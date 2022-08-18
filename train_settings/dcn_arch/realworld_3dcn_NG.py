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
# from data import processing, sampler, DataLoader
import actors.dbsr_actors as dbsr_actors
from trainers import SimpleTrainer
from utils.loading import load_network
from admin.multigpu import MultiGPU
from models.loss.image_quality_v2 import PixelWiseError, PSNR
from models.alignment.pwcnet import PWCNet
from admin.environment import env_settings

from data import DataLoader
from dataset.burstsr_dataset_bip import BurstSRDataset


def run(settings):
    settings.description = 'Default settings for fine-tuning a DBSR model on BurstSR dataset'
    settings.batch_size = 1
    crop_sz = 56
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 300

    settings.burst_sz = 14

    # data_processing_train = processing.BurstSRProcessing(transform=None, random_flip=True,
    #                                                      substract_black_level=True,
    #                                                      crop_sz=crop_sz)
    # burstsr_train = datasets.BurstSRDataset(split='train')

    # # Train sampler and loader
    # dataset_train = sampler.RandomBurst([burstsr_train], [1], burst_size=settings.burst_sz,
    #                                     samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)

    # # ********************* Val
    # data_processing_val = processing.BurstSRProcessing(transform=None,
    #                                                    substract_black_level=True, crop_sz=crop_sz)
    # burstsr_val = datasets.BurstSRDataset(split='val')

    # # Train sampler and loader
    # dataset_val = sampler.IndexedBurst([burstsr_val], burst_size=settings.burst_sz, processing=data_processing_val)

    dataset_train = BurstSRDataset(root=settings.env.burstsr_dir,  split='train', burst_size=14, crop_sz=24, random_flip=True)
    # loader_train = DataLoader(dataset_train, batch_size=settings.batch_size, shuffle=True)
    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)


    dataset_val = BurstSRDataset(root=settings.env.burstsr_dir,  split='val', burst_size=14, crop_sz=80, random_flip=False)
    # loader_val = DataLoader(dataset_val, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size)

    net = load_network('dcn_arch/synthetic_3dcn_NG')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)

    bi = 40
    objective = {
        'rgb':  PixelWiseError(metric='l1', boundary_ignore=bi), 'psnr': PSNR(boundary_ignore=40)
    }

    loss_weight = {
        'rgb': 10.0,
    }

    pwcnet = PWCNet(load_pretrained=True,
                    weights_path='{}/pwcnet-network-default.pth'.format(env_settings().pretrained_nets_dir))
    actor = dbsr_actors.DCNSRRealWorldActor(net=net, objective=objective, loss_weight=loss_weight, alignment_net=pwcnet)

    optimizer = optim.AdamW(actor.net.parameters(), lr=5e-5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-6)
    # optimizer = optim.Adam(actor.net.parameters(),lr=5e-5)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(50, load_latest=True, fail_safe=True)
