import os
import torch
import argparse
import imageio

from matplotlib import pyplot as plt
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
import cv2
from model_selection import select_model
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.visualization_utils import overlay_semantic_mask,make_sparse_matching_plot
import numpy as np
from validation.test_parser import define_model_parser
from validation.utils import matches_from_flow



def pad_to_same_shape(im1, im2):
    # pad to same shape both images with zero
    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)

    return im1, im2

def crop_image_according_to_mask(img,mask,tol = 10):
    object_mask = np.bitwise_not(np.isin(mask,[0,1,2,255]))
    h,w = mask.shape
    indices = np.argwhere(object_mask)
    y_min = max(min(indices[:,0]) - tol,0)
    y_max = min(max(indices[:,0]) + tol, h)
    x_min = max(min(indices[:,1]) - tol,0)
    x_max = min(max(indices[:, 1]) + tol, w)
    cropped = img[y_min:y_max,x_min:x_max]
    return cropped,(y_min,y_max,x_min,x_max)


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def test_model_on_image_pair(args, query_image, reference_image,contact_pixels = None):
    with torch.no_grad():
        network, estimate_uncertainty = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)

        # save original ref image shape
        ref_image_shape = reference_image.shape[:2]

        # pad both images to the same size, to be processed by network
        query_image_, reference_image_ = pad_to_same_shape(query_image, reference_image)
        # convert numpy to torch tensor and put it in right format
        query_image_ = torch.from_numpy(query_image_).permute(2, 0, 1).unsqueeze(0)
        reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)

        # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
        # specific pre-processing (/255 and rescaling) are done within the function.

        # pass both images to the network, it will pre-process the images and ouput the estimated flow
        # in dimension 1x2xHxW
        if estimate_uncertainty:
            if args.flipping_condition:
                raise NotImplementedError('No flipping condition with PDC-Net for now')

            estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_,
                                                                                              reference_image_,
                                                                                              mode='channel_first')
            confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
            confidence_map = confidence_map[:ref_image_shape[0], :ref_image_shape[1]]
        else:
            if args.flipping_condition and 'GLUNet' in args.model:
                estimated_flow = network.estimate_flow_with_flipping_condition(query_image_, reference_image_,
                                                                               mode='channel_first')
            else:
                estimated_flow = network.estimate_flow(query_image_, reference_image_, mode='channel_first')
        estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
        estimated_flow_numpy = estimated_flow_numpy[:ref_image_shape[0], :ref_image_shape[1]]
        # removes the padding

        warped_query_image = remap_using_flow_fields(query_image, estimated_flow_numpy[:, :, 0],
                                                     estimated_flow_numpy[:, :, 1]).astype(np.uint8)

        # save images
        if args.save_ind_images:
            imageio.imwrite(os.path.join(args.save_dir, 'query.png'), query_image)
            imageio.imwrite(os.path.join(args.save_dir, 'reference.png'), reference_image)
            imageio.imwrite(os.path.join(args.save_dir, 'warped_query_{}_{}.png'.format(args.model, args.pre_trained_model)),
                            warped_query_image)

        if estimate_uncertainty:
            color = [255, 102, 51]
            fig, axis = plt.subplots(1, 5, figsize=(30, 30))

            confident_mask = (confidence_map > 0.50).astype(np.uint8)
            confident_warped = overlay_semantic_mask(warped_query_image, ann=255 - confident_mask*255, color=color)
            axis[2].imshow(confident_warped)
            axis[2].set_title('Confident warped query image according to \n estimated flow by {}_{}'
                              .format(args.model, args.pre_trained_model))
            axis[4].imshow(confidence_map, vmin=0.0, vmax=1.0)
            axis[4].set_title('Confident regions')
        else:
            if not args.path_reference_object:
                fig, axis = plt.subplots(1, 4, figsize=(30, 30))
            else:
                contact_pts_matching = match_contact_points(query_image, reference_image,estimated_flow,contact_pixels)
                fig, axis = plt.subplots(1, 5, figsize=(30, 30))
            axis[2].imshow(warped_query_image)
            axis[2].set_title(
                'Warped query image according to estimated flow by {}_{}'.format(args.model, args.pre_trained_model))
        axis[0].imshow(query_image)
        axis[0].set_title('Query image')
        axis[1].imshow(reference_image)
        axis[1].set_title('Reference image')

        axis[3].imshow(flow_to_image(estimated_flow_numpy))
        axis[3].set_title('Estimated flow {}_{}'.format(args.model, args.pre_trained_model))

        if args.path_reference_object:
            axis[-1].imshow(contact_pts_matching)
            axis[-1].set_title('Contact points matching')

        fig.savefig(
            os.path.join(args.save_dir, 'Warped_query_image_{}_{}.png'.format(args.model, args.pre_trained_model)),
            bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print('Saved image!')
        return estimated_flow


def project_points_to_pixels(pts, ext_mat, int_mat):
    pts = np.array(pts)
    pts_homo = np.concatenate(
        [pts, np.ones((pts.shape[0], 1))], axis=1)  # (N,4)
    proj_mat = int_mat @ ext_mat[:3, :]
    pix_homo = proj_mat @ pts_homo.T  # (3, N)
    pixels = (pix_homo[:2, :] / pix_homo[2, :]).T  # (N, 2)

    return pixels

def match_contact_points(query_image,reference_image,estimated_flow,contact_pixels):
    mask = np.zeros(estimated_flow.shape[-2:], dtype=int)[np.newaxis, ...]
    mask_indices = contact_pixels.astype(int)[:, [1, 0]]  # (x,y) -> (row, col)
    mask[:, mask_indices[:, 0], mask_indices[:, 1]] = 1
    mask = torch.tensor(mask, device=estimated_flow.device) == 1
    # print(estimated_flow.shape)
    # print(mask.shape)
    mkpts_q, mkpts_r = matches_from_flow(estimated_flow, mask)
    # print(mkpts_q)
    # print(mkpts_r)

    confidence_values = np.ones(mkpts_q.shape[0])
    import matplotlib.cm as cm
    color = cm.jet(confidence_values)
    out = make_sparse_matching_plot(
        query_image, reference_image, mkpts_q, mkpts_r, color, margin=10)

    # plt.figure(figsize=(16, 8))
    # plt.imshow(out)
    # plt.show()
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test models on a pair of images')
    define_model_parser(parser)  # model parameters
    parser.add_argument('--pre_trained_model', type=str, help='Name of the pre-trained-model', required=True)
    parser.add_argument('--path_query_image', type=str,
                        help='Path to the source image.', required=True)
    parser.add_argument('--path_reference_image', type=str,
                        help='Path to the target image.', required=True)
    parser.add_argument('--path_query_image_mask', type=str, default = '',
                        help='Path to the source image mask.', required=False)
    parser.add_argument('--path_reference_image_mask', type=str, default = '',
                        help='Path to the target image mask.', required=False)
    parser.add_argument('--path_reference_object',type = str,default = '')
    parser.add_argument('--visualize_images',default = False,action = 'store_true')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory where to save output figure.')
    parser.add_argument('--save_ind_images', dest='save_ind_images',  default=False, type=boolean_string,
                        help='Save individual images? ')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu

    local_optim_iter = args.optim_iter if not args.local_optim_iter else int(args.local_optim_iter)

    if not os.path.exists(args.path_query_image):
        raise ValueError('The path to the source image you provide does not exist ! ')
    if not os.path.exists(args.path_reference_image):
        raise ValueError('The path to the target image you provide does not exist ! ')

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    try:
        query_image = cv2.imread(args.path_query_image, 1)[:, :, ::- 1]
        reference_image = cv2.imread(args.path_reference_image, 1)[:, :, ::- 1]

        if args.visualize_images:
            fig,axis = plt.subplots(1,2)
            axis[0].imshow(query_image)
            axis[0].set_title('Query image')
            axis[1].imshow(reference_image)
            axis[1].set_title('Reference image')
            plt.show()

        if args.path_query_image_mask:
            query_image_mask = cv2.imread(args.path_query_image_mask,cv2.IMREAD_GRAYSCALE)
            query_image,query_cropbox = crop_image_according_to_mask(query_image,query_image_mask)
        if args.path_reference_image_mask:
            reference_image_mask = cv2.imread(args.path_reference_image_mask,cv2.IMREAD_GRAYSCALE)
            reference_image,reference_cropbox = crop_image_according_to_mask(reference_image,reference_image_mask)
        if args.path_query_image_mask and args.path_reference_image_mask and args.visualize_images:
            # print(query_image.shape)
            # print(reference_image.shape)
            fig, axis = plt.subplots(1, 2)
            axis[0].imshow(query_image)
            axis[0].set_title('Query image (cropped)')
            axis[1].imshow(reference_image)
            axis[1].set_title('Reference image (cropped)')
            plt.show()

    except:
        raise ValueError('It seems that the path for the images you provided does not work ! ')

    if args.path_reference_object:
        ref_cam_idx = int(args.path_reference_image.split('.png')[0][-1])
        assert ref_cam_idx in range(4), f'Wrong cam id {ref_cam_idx}'

        demo_data = np.load(args.path_reference_object, allow_pickle=True)

        int_mats = demo_data['intrinsic_matrices']
        ext_mats = demo_data['extrinsic_matrices']

        int_mat = int_mats[ref_cam_idx]
        ext_mat = ext_mats[ref_cam_idx]
        # in pybullet format https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.cb0co8y2vuvc
        contact_pts_pybullet = demo_data['contact_points']
        contact_pts = [pts_pybullet[5] for pts_pybullet in contact_pts_pybullet]
        print(f"contact points: {contact_pts}")

    if args.path_reference_object:
        contact_pixels = project_points_to_pixels(contact_pts, ext_mat, int_mat)
        if args.path_reference_image_mask:
            contact_pixels[:, 0] -= reference_cropbox[2]
            contact_pixels[:, 1] -= reference_cropbox[0]
            if args.visualize_images:
                plt.scatter(x=contact_pixels[:, 0],
                            y=contact_pixels[:, 1], s=10, c='red', marker='o')
                plt.imshow(reference_image)
                plt.show()

    estimated_flow = test_model_on_image_pair(args, query_image, reference_image,contact_pixels)
    # if args.path_reference_object:
    #     match_contact_points(query_image, reference_image,estimated_flow,contact_pixels)



