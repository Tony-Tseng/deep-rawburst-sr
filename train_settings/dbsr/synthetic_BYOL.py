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
import models.dbsr.sslnet as sslnet
import models.dbsr.SSL_model as ssl_model
import actors.ssl_actors as ssl_actors
from trainers import SimpleTrainer
import data.transforms as tfm
from admin.multigpu import MultiGPU
from models.loss.image_quality_v2 import PSNR, CosineError, SimSiamError


def run(settings):
    settings.description = 'Default settings for training DBSR models on synthetic burst dataset '
    settings.batch_size = 8
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 500

    settings.crop_sz = (384, 384)
    settings.burst_sz = 8
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

    data_processing_train = processing.SyntheticBurstProcessing(settings.crop_sz, settings.burst_sz,
                                                                settings.downsample_factor,
                                                                burst_transformation_params=settings.burst_transformation_params,
                                                                transform=transform_train,
                                                                image_processing_params=settings.image_processing_params)
    data_processing_val = processing.SyntheticBurstProcessing(settings.crop_sz, settings.burst_sz,
                                                              settings.downsample_factor,
                                                              burst_transformation_params=settings.burst_transformation_params,
                                                              transform=transform_val,
                                                              image_processing_params=settings.image_processing_params)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([zurich_raw2rgb_train], [1],
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)
    dataset_val = sampler.RandomImage([zurich_raw2rgb_val], [1],
                                      samples_per_epoch=settings.batch_size * 200, processing=data_processing_val)

    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=5)

    net = sslnet.dbsrnet_cvpr2021(enc_init_dim=64, enc_num_res_blocks=9, enc_out_dim=512,
                                     dec_init_conv_dim=64, dec_num_pre_res_blocks=5,
                                     dec_post_conv_dim=32, dec_num_post_res_blocks=4,
                                     upsample_factor=settings.downsample_factor * 2,
                                     offset_feat_dim=64,
                                     weight_pred_proj_dim=64,
                                     num_weight_predictor_res=3,
                                     gauss_blur_sd=1.0,
                                     icnrinit=True
                                     )
    project_net = ssl_model.Projection(27648, 1024)
    predict_net = ssl_model.Predictor(27648, 1024)
    degrad_net = ssl_model.Degrade(4)
    
    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)

    objective = {'psnr': PSNR(boundary_ignore=40), 'ssl': SimSiamError(), 'cosine': CosineError()}

    loss_weight = {'rgb': 1.0}

    actor = ssl_actors.BYOLSyntheticActor(net=net, project_net=project_net, predict_net= predict_net, degrad_net= degrad_net, objective=objective, loss_weight=loss_weight)

    optimizer = optim.Adam([{'params': actor.net.parameters(), 'lr': 5e-5}],
                           lr=5e-5)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(90, load_latest=True, fail_safe=True)
 