# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

random.seed(10)
import sys
import time
sys.path.append('/home/tony/Desktop/deep-rawburst-sr/')

import data.camera_pipeline as rgb2raw
from utils.data_format_utils import torch_to_numpy, numpy_to_torch

from scipy import ndimage
import cv2
import time


def rgb2rawburst(image, burst_size, downsample_factor=1, burst_transformation_params=None,
                 image_processing_params=None, interpolation_type='bilinear'):
    """ Generates a synthetic LR RAW burst from the input image. The input sRGB image is first converted to linear
    sensor space using an inverse camera pipeline. A LR burst is then generated by applying random
    transformations defined by burst_transformation_params to the input image, and downsampling it by the
    downsample_factor. The generated burst is then mosaicekd and corrputed by random noise.

    args:
        image - input sRGB image
        burst_size - Number of images in the output burst
        downsample_factor - Amount of downsampling of the input sRGB image to generate the LR image
        burst_transformation_params - Parameters of the affine transformation used to generate a burst from single image
        image_processing_params - Parameters of the inverse camera pipeline used to obtain RAW image from sRGB image
        interpolation_type - interpolation operator used when performing affine transformations and downsampling
    """
    tmp_img = image.permute((1,2,0)).numpy()
    kernel = get_blur_kernel()[:, :, None]
    image_blur = ndimage.convolve(tmp_img, kernel, mode='reflect')
    image_blur = torch.from_numpy(image_blur).permute((2,0,1))
    
    # # debug purpose
    # save_image(tmp_img, "image")
    # save_image(image_blur, "image_blur")
    # print("save image and sleep 5 sec")
    # time.sleep(5)
    
    if image_processing_params is None:
        image_processing_params = {}

    _defaults = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True}
    for k, v in _defaults.items():
        if k not in image_processing_params:
            image_processing_params[k] = v

    # Sample camera pipeline params
    if image_processing_params['random_ccm']:
        rgb2cam = rgb2raw.random_ccm()
    else:
        rgb2cam = torch.eye(3).float()
    cam2rgb = rgb2cam.inverse()

    # Sample gains
    if image_processing_params['random_gains']:
        rgb_gain, red_gain, blue_gain = rgb2raw.random_gains()
    else:
        rgb_gain, red_gain, blue_gain = (1.0, 1.0, 1.0)

    # print("syn: ", type(image))
    # Approximately inverts global tone mapping.
    use_smoothstep = image_processing_params['smoothstep']
    if use_smoothstep:
        image = rgb2raw.invert_smoothstep(image)
        image_blur = rgb2raw.invert_smoothstep(image_blur)

    # Inverts gamma compression.
    use_gamma = image_processing_params['gamma']
    if use_gamma:
        image = rgb2raw.gamma_expansion(image)
        image_blur = rgb2raw.gamma_expansion(image_blur)

    # Inverts color correction.
    image = rgb2raw.apply_ccm(image, rgb2cam)
    image_blur = rgb2raw.apply_ccm(image_blur, rgb2cam)

    # Approximately inverts white balance and brightening.
    image = rgb2raw.safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    image_blur = rgb2raw.safe_invert_gains(image_blur, rgb_gain, red_gain, blue_gain)

    # Clip saturated pixels.
    image = image.clamp(0.0, 1.0)
    # image_blur = np.clip(image_blur, 0.0, 1.0)
    image_blur = image_blur.clamp(0.0, 1.0)

    # Generate LR burst
    image_burst_rgb, flow_vectors = single2lrburst(image, burst_size=burst_size,
                                                   downsample_factor=downsample_factor,
                                                   transformation_params=burst_transformation_params,
                                                   interpolation_type=interpolation_type)
    
    if burst_transformation_params.get('border_crop') is not None:
        border_crop = burst_transformation_params.get('border_crop')
        image_blur = image_blur[:, border_crop:-border_crop, border_crop:-border_crop]
    
    image_blur = image_blur.permute((1,2,0)).numpy()
    blur_down = cv2.resize(image_blur, None, fx=1.0 / downsample_factor, fy=1.0 / downsample_factor,
                             interpolation=cv2.INTER_LINEAR)
    blur_down = torch.from_numpy(blur_down).permute((2,0,1))
    blur_raw = rgb2raw.mosaic(blur_down)
    if image_processing_params['add_noise']:
        shot_noise_level, read_noise_level = rgb2raw.random_noise_levels()
        blur_raw = rgb2raw.add_noise(blur_raw, shot_noise_level, read_noise_level)
    blur_raw = blur_raw.clamp(0.0, 1.0)
    
    plot = False
    if plot:
        time.sleep(5)
        num = np.random.randint(1000000)
        plt.figure(num)
        flow = flow_vectors[..., 50, 50].numpy()
        for f in flow:
            plt.arrow(0, 0, f[0], f[1], color='red', width = 0.01, head_width = 0.1)
        
        # a = burst_transformation_params["a"]
        # b = burst_transformation_params["b"]
        # theta = burst_transformation_params["theta"]
        
        # for i in range(100):
        #     phi = np.random.uniform(0, 1)*2*np.pi
        #     x_e = np.cos(phi) * a/4
        #     y_e = np.sin(phi) * b/4
        #     translation = (x_e*np.cos(theta) - y_e*np.sin(theta), 
        #                     x_e*np.sin(theta) + y_e*np.cos(theta))
        #     plt.scatter(translation[0], translation[1], color='black', s=0.5)
            
        plt.savefig(f"sample_plot/uniform/sample_{num}.png")
    

    # mosaic
    image_burst = rgb2raw.mosaic(image_burst_rgb.clone())

    # Add noise
    if image_processing_params['add_noise']:
        shot_noise_level, read_noise_level = rgb2raw.random_noise_levels()
        image_burst = rgb2raw.add_noise(image_burst, shot_noise_level, read_noise_level)
    else:
        shot_noise_level = 0
        read_noise_level = 0

    # Clip saturated pixels.
    image_burst = image_burst.clamp(0.0, 1.0)

    meta_info = {'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain, 'red_gain': red_gain,
                 'blue_gain': blue_gain, 'smoothstep': use_smoothstep, 'gamma': use_gamma,
                 'shot_noise_level': shot_noise_level, 'read_noise_level': read_noise_level}
    return image_burst, image, image_burst_rgb, flow_vectors, meta_info, blur_raw


def get_tmat(image_shape, translation, theta, shear_values, scale_factors):
    """ Generates a transformation matrix corresponding to the input transformation parameters """
    im_h, im_w = image_shape

    t_mat = np.identity(3)

    t_mat[0, 2] = translation[0]
    t_mat[1, 2] = translation[1]
    t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
    t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

    t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                        [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                        [0.0, 0.0, 1.0]])

    t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                        [0.0, scale_factors[1], 0.0],
                        [0.0, 0.0, 1.0]])

    t_mat = t_scale @ t_rot @ t_shear @ t_mat
    t_mat = t_mat[:2, :]

    return t_mat


def single2lrburst(image, burst_size, downsample_factor=1, transformation_params=None,
                   interpolation_type='bilinear'):
    """ Generates a burst of size burst_size from the input image by applying random transformations defined by
    transformation_params, and downsampling the resulting burst by downsample_factor.

    args:
        image - input sRGB image
        burst_size - Number of images in the output burst
        downsample_factor - Amount of downsampling of the input sRGB image to generate the LR image
        transformation_params - Parameters of the affine transformation used to generate a burst from single image
        interpolation_type - interpolation operator used when performing affine transformations and downsampling
    """

    if interpolation_type == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation_type == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError

    # Convert float array to uint8 format
    normalize = False
    if isinstance(image, torch.Tensor):
        if image.max() < 2.0:
            image = image * 255.0
            normalize = True
        image = torch_to_numpy(image).astype(np.uint8)

    burst = []
    sample_pos_inv_all = []

    rvs, cvs = torch.meshgrid([torch.arange(0, image.shape[0]),
                               torch.arange(0, image.shape[1])])
    
    # print(rvs.shape) -> torch.Size([432, 432])
    # print(cvs.shape) -> torch.Size([432, 432])
    sample_grid = torch.stack((cvs, rvs, torch.ones_like(cvs)), dim=-1).float()
    
    # print(sample_grid.shape) -> torch.Size([432, 432, 3])

    for i in range(burst_size):
        if i == 0:
            # For base image, do not apply any random transformations. We only translate the image to center the
            # sampling grid
            shift = (downsample_factor / 2.0) - 0.5
            translation = (shift, shift)
            theta = 0.0
            shear_factor = (0.0, 0.0)
            scale_factor = (1.0, 1.0)
        else:
            # Sample random image transformation parameters
            # max_translation = 24
            max_translation = transformation_params.get('max_translation', 0.0)

            if max_translation <= 0.01:
                shift = (downsample_factor / 2.0) - 0.5
                translation = (shift, shift)
            elif "ellipse" in transformation_params.keys() and transformation_params["ellipse"]==True:
                a = transformation_params.get('a', 1.0)
                b = transformation_params.get('b', 1.0)
                theta_ellipse = transformation_params.get('theta', 0.0)
                phi = np.random.uniform(0, 1)*2*np.pi
                x_e = np.cos(phi) * a
                y_e = np.sin(phi) * b
                translation = (x_e*np.cos(theta_ellipse) - y_e*np.sin(theta_ellipse), 
                               x_e*np.sin(theta_ellipse) + y_e*np.cos(theta_ellipse))
                # print((translation[0])**2/a**2 + (translation[1])**2/b**2)
            else:
                translation = (random.uniform(-max_translation, max_translation),
                               random.uniform(-max_translation, max_translation))

            max_rotation = transformation_params.get('max_rotation', 0.0)
            theta = random.uniform(-max_rotation, max_rotation)

            # Skew
            max_shear = transformation_params.get('max_shear', 0.0)
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            shear_factor = (shear_x, shear_y)

            max_ar_factor = transformation_params.get('max_ar_factor', 0.0)
            ar_factor = np.exp(random.uniform(-max_ar_factor, max_ar_factor))

            max_scale = transformation_params.get('max_scale', 0.0)
            scale_factor = np.exp(random.uniform(-max_scale, max_scale))

            scale_factor = (scale_factor, scale_factor * ar_factor)

        output_sz = (image.shape[1], image.shape[0])

        # Generate a affine transformation matrix corresponding to the sampled parameters
        t_mat = get_tmat((image.shape[0], image.shape[1]), translation, theta, shear_factor, scale_factor)
        t_mat_tensor = torch.from_numpy(t_mat)

        # Apply the sampled affine transformation
        
        # Affine images translate by t_mat
        image_t = cv2.warpAffine(image, t_mat, output_sz, flags=interpolation,
                                 borderMode=cv2.BORDER_CONSTANT)

        t_mat_tensor_3x3 = torch.cat((t_mat_tensor.float(), torch.tensor([0.0, 0.0, 1.0]).view(1, 3)), dim=0)
        t_mat_tensor_inverse = t_mat_tensor_3x3.inverse()[:2, :].contiguous()

        # Calculate the position vector to origin base image by inverse of t_mat
        sample_pos_inv = torch.mm(sample_grid.view(-1, 3), t_mat_tensor_inverse.t().float()).view(
            *sample_grid.shape[:2], -1)

        if transformation_params.get('border_crop') is not None:
            border_crop = transformation_params.get('border_crop')

            image_t = image_t[border_crop:-border_crop, border_crop:-border_crop, :]
            sample_pos_inv = sample_pos_inv[border_crop:-border_crop, border_crop:-border_crop, :]

        # Downsample the image
        image_t = cv2.resize(image_t, None, fx=1.0 / downsample_factor, fy=1.0 / downsample_factor,
                             interpolation=interpolation)
        sample_pos_inv = cv2.resize(sample_pos_inv.numpy(), None, fx=1.0 / downsample_factor,
                                    fy=1.0 / downsample_factor,
                                    interpolation=interpolation)

        sample_pos_inv = torch.from_numpy(sample_pos_inv).permute(2, 0, 1)

        if normalize:
            image_t = numpy_to_torch(image_t).float() / 255.0
        else:
            image_t = numpy_to_torch(image_t).float()
        burst.append(image_t)
        sample_pos_inv_all.append(sample_pos_inv / downsample_factor)

    burst_images = torch.stack(burst)
    sample_pos_inv_all = torch.stack(sample_pos_inv_all)

    # Compute the flow vectors to go from the i'th burst image to the base image
    flow_vectors = sample_pos_inv_all - sample_pos_inv_all[:1, ...]

    return burst_images, flow_vectors


def get_blur_kernel(train=True):
    if train:
        gaussian_sigma = random.choice(
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    else:
        gaussian_sigma = 2.0
    gaussian_blur_kernel_size = 13
    kernel = matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
    return kernel

def matlab_style_gauss2D(shape=(5, 5), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def save_image(img, name):
    img = np.round(255.0 * np.clip(img, 0.0, 1.0)).astype(np.uint8)
    cv2.imwrite(f'{name}.png', img)
    

if __name__ == '__main__':
    image = np.load("/home/tony/Desktop/deep-rawburst-sr/data/image.npy")
    image = torch.from_numpy(image)
    # print(image.shape) -> torch.Size([3, 432, 432])
    
    burst_size = 8
    downsample_factor = 4
    a, b = random.uniform(0, 24), random.uniform(0, 24)
    theta = random.uniform(0, 180) * np.pi / 180
    burst_transformation_params = {'max_translation': 1,
                                    'max_rotation': 0.0,
                                    'max_shear': 0.0,
                                    'max_scale': 0.0,
                                    'border_crop': 24,
                                    'ellipse': True,
                                    'theta': theta,
                                    'a': a,
                                    'b': b}
    
    image_burst_rgb, flow_vectors = single2lrburst(image, burst_size=burst_size,
                                                   downsample_factor=downsample_factor,
                                                   transformation_params=burst_transformation_params)
    # (432 - max_translation*2) / 4 = 96
    # print(image_burst_rgb.shape) -> torch.Size([8, 3, 96, 96])
    # print(flow_vectors.shape) -> torch.Size([8, 2, 96, 96])
    
    # for flow in flow_vectors:
    flow = flow_vectors[..., 50, 50].numpy()

    for f in flow:
        plt.arrow(0, 0, f[0], f[1], color='red', width = 0.01, head_width = 0.1)
    
    for i in range(100):    
        phi = np.random.uniform(0, 1)*2*np.pi
        x_e = np.cos(phi) * a/4
        y_e = np.sin(phi) * b/4
        translation = (x_e*np.cos(theta) - y_e*np.sin(theta), 
                        x_e*np.sin(theta) + y_e*np.cos(theta))
        plt.scatter(translation[0], translation[1], color='black', s=0.5)
    
    plt.savefig("sample_dir.png")