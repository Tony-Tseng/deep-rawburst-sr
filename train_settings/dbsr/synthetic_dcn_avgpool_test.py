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

import torch.optim as optim
import dataset as datasets
from data import processing, sampler, DataLoader
# import models.dbsr.dbsrnet as dbsr_nets
import models.DCN.dcnsr_avgpool_test as dcnsr_net
import actors.dbsr_actors as dbsr_actors
from trainers import SimpleTrainer
import data.transforms as tfm
from admin.multigpu import *
from models.loss.image_quality_v2 import PSNR, PixelWiseError
import torch.nn as nn

from dataset.synthetic_burst_train_set import SyntheticBurst

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run(settings):
    settings.description = 'Default settings for training DBSR models on synthetic burst dataset '
    settings.batch_size = 1
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 1000

    settings.crop_sz = (384, 384)
    settings.burst_sz = 14
    settings.downsample_factor = 4

    settings.burst_transformation_params = {'max_translation': 24.0,
                                            'max_rotation': 1.0,
                                            'max_shear': 0.0,
                                            'max_scale': 0.0,
                                            'border_crop': 24}
    settings.burst_reference_aligned = True
    settings.image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True}

    zurich_raw2rgb_train = datasets.ZurichRAW2RGB(split='train')
    zurich_raw2rgb_val = datasets.ZurichRAW2RGB(split='test')

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())

    # data_processing_train = processing.SyntheticBurstProcessing(settings.crop_sz, settings.burst_sz,
    #                                                             settings.downsample_factor,
    #                                                             burst_transformation_params=settings.burst_transformation_params,
    #                                                             transform=transform_train,
    #                                                             image_processing_params=settings.image_processing_params)
    # data_processing_val = processing.SyntheticBurstProcessing(settings.crop_sz, settings.burst_sz,
    #                                                           settings.downsample_factor,
    #                                                           burst_transformation_params=settings.burst_transformation_params,
    #                                                           transform=transform_val,
    #                                                           image_processing_params=settings.image_processing_params)

    # Train sampler and loader
    # train_dataset = sampler.RandomImage([zurich_raw2rgb_train], [1],
    #                                     samples_per_epoch=settings.batch_size * 5000, processing=data_processing_train)
    # test_dataset = sampler.RandomImage([zurich_raw2rgb_val], [1],
    #                                   samples_per_epoch=settings.batch_size * 300, processing=data_processing_val)
    
    train_dataset = SyntheticBurst(zurich_raw2rgb_train, burst_size=settings.burst_sz, crop_sz=384, transform=transform_train)    
    test_dataset = SyntheticBurst(zurich_raw2rgb_val, burst_size=settings.burst_sz, crop_sz=384, transform=transform_val)

    loader_train = DataLoader('train', train_dataset, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', test_dataset, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=5)

    net = dcnsr_net.dcnsrnet_unet_mergediff(alignment_init_dim=64, reduction=8, alignment_out_dim=64, 
                             dec_init_conv_dim=64, dec_num_pre_res_blocks=5, dec_post_conv_dim=32, 
                             dec_num_post_res_blocks=4, burst_size=settings.burst_sz, upsample_factor=settings.downsample_factor * 2, 
                             icnrinit=True, gauss_blur_sd=1.0)

    total_param = count_parameters(net)
    print("The total Net parameter is ", total_param)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        # net = MultiGPU(net, dim=0)
        net = nn.DataParallel(net, device_ids=[0,1,2])

    objective = {'rgb': PixelWiseError(metric='l1', boundary_ignore=40), 'psnr': PSNR(boundary_ignore=40)}

    loss_weight = {'rgb': 1.0}

    actor = dbsr_actors.DCNSRSyntheticActor(net=net, objective=objective, loss_weight=loss_weight)

    optimizer = optim.AdamW(actor.net.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-6)
    
    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(150, load_latest=True, fail_safe=True)
