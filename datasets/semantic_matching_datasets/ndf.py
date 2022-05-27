from __future__ import print_function, division
import os
import torch
import cv2
import pandas as pd
import numpy as np
from datasets.util import pad_to_same_shape
from .semantic_keypoints_datasets import SemanticKeypointsDataset, random_crop, random_crop_image
from ..util import pad_to_same_shape, pad_to_size, resize_keeping_aspect_ratio
from datasets.util import define_mask_zero_borders
import scipy.io as sio
import random


def read_mat(path, obj_name):
    r"""Reads specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj


def crop_image(img, bbox, enlarge=20):
    h, w = img.shape[:2]
    y_min = max(bbox[1].int() - enlarge, 0)
    y_max = min(bbox[3].int() + enlarge, h)
    x_min = max(bbox[0].int() - enlarge, 0)
    x_max = min(bbox[2].int() + enlarge, w)
    cropped = img[y_min:y_max, x_min:x_max]
    bbox_new = torch.tensor([bbox[0] - x_min, bbox[1] - y_min, x_max - bbox[2], y_max - bbox[3]]).float()
    return cropped,bbox_new


class NDFDataset(SemanticKeypointsDataset):
    """
    Proposal Flow image pair dataset (PF-Pascal).
    There is a certain number of pairs per category and the number of keypoints per pair also varies
    """

    def __init__(self, root, split, thres='img', annotated=False, pre_cropped=True, source_image_transform=None,
                 target_image_transform=None, flow_transform=None, output_image_size=None, training_cfg=None):
        """
        Args:
            root:
            split: 'test', 'val', 'train'
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            output_image_size: size if images and annotations need to be resized, used when split=='test'
            training_cfg: training config
        Output in __getittem__  (for split=='test'):
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse)
            source_kps
            target_kps
        """
        super(NDFDataset, self).__init__('ndf', root, thres, split, source_image_transform,
                                         target_image_transform, flow_transform, training_cfg=training_cfg,
                                         output_image_size=output_image_size)

        # TODO: generate csv for training
        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.cls = ['mug', 'bowl', 'bottle']
        self.cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1

        # if split == 'train':
        #     self.flip = self.train_data.iloc[:, 3].values.astype('int') # seems useless
        self.src_kps = []
        self.trg_kps = []
        self.src_bbox = []
        self.trg_bbox = []
        self.annotated = annotated  # whether we have keypoint info
        self.pre_cropped = pre_cropped  # whether to crop image first

        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            src_mask_path = os.path.join(self.ann_path, os.path.basename(src_imname).replace('rgb', 'mask'))
            trg_mask_path = os.path.join(self.ann_path, os.path.basename(trg_imname).replace('rgb', 'mask'))
            src_bbox = self.get_bbox_from_mask(src_mask_path)
            trg_bbox = self.get_bbox_from_mask(trg_mask_path)
            self.src_bbox.append(src_bbox)
            self.trg_bbox.append(trg_bbox)

        if self.annotated:
            # here reads bounding box and keypoints information from annotation files. Also in most of the csv files.
            for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
                src_anns = os.path.join(self.ann_path, self.cls[cls],
                                        os.path.basename(src_imname))[:-4] + '.mat'
                trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                        os.path.basename(trg_imname))[:-4] + '.mat'

                src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
                trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()

                src_kps = []
                trg_kps = []
                for src_kk, trg_kk in zip(src_kp, trg_kp):
                    if torch.isnan(src_kk).sum() > 0 or torch.isnan(trg_kk).sum() > 0:
                        continue
                    else:
                        src_kps.append(src_kk)
                        trg_kps.append(trg_kk)
                self.src_kps.append(torch.stack(src_kps).t())
                self.trg_kps.append(torch.stack(trg_kps).t())

        # self._imnames are basenames
        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.basename(x), self.trg_imnames))

        # if need to resize the images, even for testing
        if output_image_size is not None:
            if not isinstance(output_image_size, tuple):
                output_image_size = (output_image_size, output_image_size)
        self.output_image_size = output_image_size

    def __getitem__(self, idx):
        """
        Args:
            idx:

        Returns: If split is test, dictionary with fieldnames:
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse)
            source_kps
            target_kps
        """
        batch = super(NDFDataset, self).__getitem__(idx)

        batch['sparse'] = True
        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize_ori'])
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize_ori'])

        if self.pre_cropped:
            batch['source_image'],batch['src_bbox'] = \
                crop_image(batch['source_image'],batch['src_bbox'])
            batch['source_image'],batch['trg_bbox'] = \
                crop_image(batch['target_image'],batch['trg_bbox'])
            batch['source_image_size'] = np.array(batch['source_image'].shape[:2])  # h,w
            batch['target_image_size'] = np.array(batch['target_image'].shape[:2])

            # TODO: how to modify batch['src_imsize_ori'] and batch['trgs_imsize_ori']

            if self.annotated:
                raise NotImplementedError


        if self.split != 'test':

            if self.training_cfg['augment_with_crop']:
                if self.annotated:
                    batch['source_image'], batch['source_kps'], batch['src_bbox'] = random_crop(
                        batch['source_image'], batch['source_kps'].clone(), batch['src_bbox'].int(),
                        size=self.training_cfg['crop_size'], p=self.training_cfg['proba_of_crop'])

                    batch['target_image'], batch['target_kps'], batch['trg_bbox'] = random_crop(
                        batch['target_image'], batch['target_kps'].clone(), batch['trg_bbox'].int(),
                        size=self.training_cfg['crop_size'], p=self.training_cfg['proba_of_crop'])
                else:
                    batch['source_image'], batch['src_bbox'] = random_crop_image(
                        batch['source_image'], batch['src_bbox'].int(), size=self.training_cfg['crop_size'],
                        p=self.training_cfg['proba_of_crop'])

                    batch['target_image'], batch['trg_bbox'] = random_crop_image(
                        batch['target_image'], batch['trg_bbox'].int(), size=self.training_cfg['crop_size'],
                        p=self.training_cfg['proba_of_crop'])

            if self.training_cfg['augment_with_flip']:
                if self.annotated:
                    if random.random() < self.training_cfg['proba_of_batch_flip']:
                        self.horizontal_flip(batch)
                    else:
                        if random.random() < self.training_cfg['proba_of_image_flip']:
                            batch['source_image'], batch['src_bbox'], batch['source_kps'] = self.horizontal_flip_img(
                                batch['source_image'], batch['src_bbox'], batch['source_kps'])
                        if random.random() < self.training_cfg['proba_of_image_flip']:
                            batch['target_image'], batch['trg_bbox'], batch['target_kps'] = self.horizontal_flip_img(
                                batch['target_image'], batch['trg_bbox'], batch['target_kps'])
                else:
                    self.horizontal_flip_img_bbox(batch)

            '''
            # Horizontal flipping of both images and key-points during training
            if self.split == 'train' and self.flip[idx]:
                self.horizontal_flip(batch)
                batch['flip'] = 1
            else:
                batch['flip'] = 0
            '''

            batch = self.recover_image_pair_for_training(batch) if self.annotated \
                else self.recover_image_pair_for_training_only_image(batch)

            batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize_ori'],
                                              output_image_size=self.training_cfg['output_image_size'])
            batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize_ori'],
                                              output_image_size=self.training_cfg['output_image_size'])

            batch['pckthres'] = self.get_pckthres(batch, batch['source_image_size'])

            if self.source_image_transform is not None:
                batch['source_image'] = self.source_image_transform(batch['source_image'])
            if self.target_image_transform is not None:
                batch['target_image'] = self.target_image_transform(batch['target_image'])

            if self.annotated:
                flow = batch['flow_map']
                if self.flow_transform is not None and flow is not None:
                    if type(flow) in [tuple, list]:
                        # flow field at different resolution
                        for i in range(len(flow)):
                            flow[i] = self.flow_transform(flow[i])
                    else:
                        flow = self.flow_transform(flow)
                batch['flow_map'] = flow

            if self.training_cfg['compute_mask_zero_borders']:
                mask_valid = define_mask_zero_borders(batch['target_image'])
                batch['mask_zero_borders'] = mask_valid

        else:
            batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize_ori'],
                                              output_image_size=self.output_image_size)
            batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize_ori'],
                                              output_image_size=self.output_image_size)

            batch['pckthres'] = self.get_pckthres(batch, batch['source_image_size'])

            batch['source_image'], batch['target_image'] = pad_to_same_shape(batch['source_image'],
                                                                             batch['target_image'])
            h_size, w_size, _ = batch['target_image'].shape

            if self.annotated:
                flow, mask = self.keypoints_to_flow(batch['source_kps'][:batch['n_pts']],
                                                    batch['target_kps'][:batch['n_pts']],
                                                    h_size=h_size, w_size=w_size)
                if self.flow_transform is not None:
                    flow = self.flow_transform(flow)
                batch['flow_map'] = flow
                batch['correspondence_mask'] = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()

            if self.source_image_transform is not None:
                batch['source_image'] = self.source_image_transform(batch['source_image'])
            if self.target_image_transform is not None:
                batch['target_image'] = self.target_image_transform(batch['target_image'])

        return batch

    def get_bbox(self, bbox_list, idx, original_image_size=None, output_image_size=None):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        if self.output_image_size is not None or output_image_size is not None:
            if output_image_size is None:
                bbox[0::2] *= (self.output_image_size[1] / original_image_size[1])  # w
                bbox[1::2] *= (self.output_image_size[0] / original_image_size[0])
            else:
                bbox[0::2] *= (float(output_image_size[1]) / float(original_image_size[1]))
                bbox[1::2] *= (float(output_image_size[0]) / float(original_image_size[0]))
        return bbox

    def get_bbox_from_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        object_mask = np.bitwise_not(np.isin(mask, [0, 1, 2, 255]))
        h, w = mask.shape
        indices = np.argwhere(object_mask)
        y_min = max(min(indices[:, 0]), 0)
        y_max = min(max(indices[:, 0]), h)
        x_min = max(min(indices[:, 1]), 0)
        x_max = min(max(indices[:, 1]), w)
        bbox = torch.tensor([x_min, y_min, x_max, y_max]).float()
        return bbox

    def horizontal_flip_img_bbox(self, batch):
        tmp = batch['src_bbox'][0].clone()
        batch['src_bbox'][0] = batch['source_image'].shape[1] - batch['src_bbox'][2]
        batch['src_bbox'][2] = batch['source_image'].shape[1] - tmp

        tmp = batch['trg_bbox'][0].clone()
        batch['trg_bbox'][0] = batch['target_image'].shape[1] - batch['trg_bbox'][2]
        batch['trg_bbox'][2] = batch['target_image'].shape[1] - tmp

        batch['source_image'] = np.flip(batch['source_image'], 1)
        batch['target_image'] = np.flip(batch['target_image'], 1)

    def recover_image_pair_for_training_only_image(self, batch):

        source = batch['source_image']
        target = batch['target_image']

        if self.training_cfg['pad_to_same_shape']:
            # either pad to same shape
            source, target = pad_to_same_shape(source, target)

        if self.training_cfg['output_image_size'] is not None:
            if isinstance(self.training_cfg['output_image_size'], list):
                # resize to a fixed load_size and rescale the keypoints accordingly
                h1, w1 = source.shape[:2]
                source = cv2.resize(source, (self.training_cfg['output_image_size'][1],
                                             self.training_cfg['output_image_size'][0]))

                h2, w2 = target.shape[:2]
                target = cv2.resize(target, (self.training_cfg['output_image_size'][1],
                                             self.training_cfg['output_image_size'][0]))
            else:
                # rescale both images so that the largest dimension is equal to the desired load_size of image and
                # then pad to obtain load_size 256x256 or whatever desired load_size. and change keypoints accordingly
                source, ratio_1 = resize_keeping_aspect_ratio(source, self.training_cfg['output_image_size'])
                source = pad_to_size(source, self.training_cfg['output_image_size'])

                target, ratio_2 = resize_keeping_aspect_ratio(target, self.training_cfg['output_image_size'])
                target = pad_to_size(target, self.training_cfg['output_image_size'])

        batch['source_image'] = source
        batch['target_image'] = target
        batch['source_image_size'] = np.array(source.shape[:2])
        batch['target_image_size'] = np.array(target.shape[:2])

        return batch
