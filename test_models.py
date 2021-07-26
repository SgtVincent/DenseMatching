import os
import torch
import argparse
import imageio
from matplotlib import pyplot as plt
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
import cv2
from model_selection import select_model
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.visualization_utils import overlay_semantic_mask
import numpy as np
from validation.test_parser import define_pdcnet_parser


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


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def test_model_on_image_pair(args, query_image, reference_image):
    with torch.no_grad():
        network, estimate_uncertainty = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.pre_trained_models_dir)

        # convert numpy to torch tensor and put it in right shape
        query_image_ = torch.from_numpy(query_image).permute(2, 0, 1).unsqueeze(0)
        reference_image_ = torch.from_numpy(reference_image).permute(2, 0, 1).unsqueeze(0)
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
        else:
            if args.flipping_condition and 'GLUNet' in args.model:
                estimated_flow = network.estimate_flow_with_flipping_condition(query_image_, reference_image_,
                                                                               mode='channel_first')
            else:
                estimated_flow = network.estimate_flow(query_image_, reference_image_, mode='channel_first')
        estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
        warped_query_image = remap_using_flow_fields(query_image, estimated_flow.squeeze()[0].cpu().numpy(),
                                                     estimated_flow.squeeze()[1].cpu().numpy()).astype(np.uint8)

        # save images
        '''
        imageio.imwrite(os.path.join(args.write_dir, 'query.png'), query_image)
        imageio.imwrite(os.path.join(args.write_dir, 'reference.png'), reference_image)
        imageio.imwrite(os.path.join(args.write_dir, 'warped_query_{}_{}.png'.format(args.model, args.pre_trained_model)),
                        warped_query_image)
        '''
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
            fig, axis = plt.subplots(1, 4, figsize=(30, 30))
            axis[2].imshow(warped_query_image)
            axis[2].set_title(
                'Warped query image according to estimated flow by {}_{}'.format(args.model, args.pre_trained_model))
        axis[0].imshow(query_image)
        axis[0].set_title('Query image')
        axis[1].imshow(reference_image)
        axis[1].set_title('Reference image')

        axis[3].imshow(flow_to_image(estimated_flow_numpy))
        axis[3].set_title('Estimated flow {}_{}'.format(args.model, args.pre_trained_model))
        fig.savefig(
            os.path.join(args.write_dir, 'Warped_query_image_{}_{}.png'.format(args.model, args.pre_trained_model)),
            bbox_inches='tight')
        plt.close(fig)
        print('Saved image!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test models on a pair of images')
    parser.add_argument('--model', type=str, help='Model to use', required=True)
    parser.add_argument('--flipping_condition', dest='flipping_condition',  default=False, type=boolean_string,
                        help='Apply flipping condition for semantic data and GLU-Net-based networks ? ')
    parser.add_argument('--optim_iter', type=int, default=3,
                        help='Number of optim iter for Global GOCor, if applicable')
    parser.add_argument('--local_optim_iter', dest='local_optim_iter', default=None,
                        help='Number of optim iter for Local GOCor, if applicable')
    parser.add_argument('--pre_trained_models_dir', type=str, default='pre_trained_models/',
                        help='Directory containing the pre-trained-models.')
    parser.add_argument('--pre_trained_model', type=str, help='Name of the pre-trained-model', required=True)
    parser.add_argument('--path_query_image', type=str,
                        help='Path to the source image.', required=True)
    parser.add_argument('--path_reference_image', type=str,
                        help='Path to the target image.', required=True)
    parser.add_argument('--write_dir', type=str,
                        help='Directory where to write output figure.')
    subparsers = parser.add_subparsers(dest='network_type')
    define_pdcnet_parser(subparsers)
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

    if not os.path.isdir(args.write_dir):
        os.makedirs(args.write_dir)
    try:
        query_image = imageio.imread(args.path_query_image)
        reference_image = imageio.imread(args.path_reference_image)
        query_image, reference_image = pad_to_same_shape(query_image, reference_image)
    except:
        raise ValueError('It seems that the path for the images you provided does not work ! ')

    test_model_on_image_pair(args, query_image, reference_image)

