import argparse
import torch
import torch.utils.data
from multiprocessing import Manager

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as d
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


import utils
import color_flow
import loaders.SceneFlowFilenames
from loaders.SceneFlowLoader import SceneFlowLoader
from models.ElParoFlowNetModel import ElParoFlowNet
from models.ElParoFlowNetModelS import ElParoFlowNetS
from models.ElParoFlowNetModelCvA import ElParoFlowNetCvA
from models.ElParoFlowNetModelCvB import ElParoFlowNetCvB
from models.ElParoFlowNetModelCvC import ElParoFlowNetCvC
from models.ElParoFlowNetModelCvD import ElParoFlowNetCvD
from models.ElParoFlowNetModelCvE import ElParoFlowNetCvE
from models.ElParoFlowNetModelCvCi import ElParoFlowNetCvCi
from models.ElParoFlowNetModelCvEa import ElParoFlowNetCvEa
from models.ElParoFlowNetModelCvEb import ElParoFlowNetCvEb
from models.ElParoFlowNetModelCvEc import ElParoFlowNetCvEc
from models.ElParoFlowNetModelCvEd import ElParoFlowNetCvEd
from models.ElParoFlowNetModelCvEcA import ElParoFlowNetCvEcA
from models.ElParoFlowNetModelCvEcB import ElParoFlowNetCvEcB
from models.ElParoFlowNetModelCvEcC import ElParoFlowNetCvEcC
from models.ElParoFlowNetModelCvEcD import ElParoFlowNetCvEcD
from models.FlowNetC import FlowNetC
from models.FlowNetS import FlowNetS
from models.FlowNetPWC import FlowNetPwcLike
from models.FlowNetPWCLite import FlowNetPwcLite
from models.FlowNetPWS import FlowNetPws
from models.FlowNetPWSDeep import FlowNetPwsDeep
from models.FlowNetPWCLite2x import FlowNetPwcLite2x
from models.FlowNetPWCLite2xA import FlowNetPwcLite2xA
from models.FlowNetPWC3x import FlowNetPwc3x

from models.Trainable import Trainable

argParser = argparse.ArgumentParser(description="El Paro Net...")
argParser.add_argument('--epochs', type=int, default=100, help='number of epochs do')
argParser.add_argument('--batch_size', type=int, default=16, help='training batch size')
argParser.add_argument('--log_every', type=int, default=100, help='Log losses after every n mini batches')
argParser.add_argument('--scene_flow_dir', type=str, required=True, help='dir with SceneFlow dataset in')
argParser.add_argument('--save_path', type=str, default='checkpoints', help='dir to save model during training.')

argParser.add_argument('--test', type=bool, default=False, help='Testing not training')
args = argParser.parse_args()

log = utils.logger("EPFlowNet")


def train(loader, trainables):
    loader_size = len(loader)
    scale = 15

    for trainable in trainables:
        trainable.train()

    for batch_index, (image_a, image_b, flow_a2b_gt, flow_b2a_gt) in enumerate(loader):
        # if batch_index > 10:
        #     return True

        (batch_size, image_c, image_h, image_w) = image_a.size()
        # image_a = image_a.cuda()
        # image_b = image_b.cuda()
        # flow_a2b_gt = flow_a2b_gt.cuda()
        #flow_b2a_gt = flow_b2a_gt.cuda()

        for trainable in trainables:
            trainable.zero_grad()

            # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
            #     with profiler.record_function("model_inference"):
            flow_prediction = trainable(image_a, image_b)

            # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))
            # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
            # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

            # fig = plt.figure(figsize=(18, 12), dpi=112)
            # ax1 = fig.add_subplot(221)
            # ax1.imshow(image_a[0].squeeze(0).cpu(), cmap='gray')
            # ax2 = fig.add_subplot(222)
            # ax2.imshow(image_b[0].squeeze(0).cpu(), cmap='gray')
            # ax3 = fig.add_subplot(223)
            # # ax3.hist(torch.flatten(flow_a2b_gt_cropped_x15[0][0].detach().cpu()).numpy(), bins=100)
            # ax3.imshow(color_flow.flow_to_rgb(flow_a2b_gt[0].detach().cpu().squeeze(0), rad_normaliser=100))
            # ax4 = fig.add_subplot(224)
            # # ax4.hist(torch.flatten(flow_a2b_gt_cropped_x15[0][1].detach().cpu()).numpy(), bins=100)
            # ax4.imshow(color_flow.flow_to_rgb(flow_prediction[0][0].detach().cpu().squeeze(0), rad_normaliser=100))
            # plt.show()

            loss, epe = trainable.loss(flow_prediction, flow_a2b_gt)
            # mask = torch.ones_like(flow_a2b_gt, dtype=bool)
            # count = mask[:, 0, :, :].sum(dim=2).sum(dim=1)
            # difference = flow_a2b_gt - flow_prediction
            # per_pixel_losses = torch.sqrt(torch.sum(torch.pow(difference, 2), dim=1))
            # mask_size = mask.size()
            # loss = torch.sum(per_pixel_losses) / (mask_size[0] * mask_size[2] * mask_size[3])
            loss.backward()
            trainable.step()
            loss = loss.detach().item()
            epe = epe.detach().item()
            trainable.update_batch_loss(loss, epe)

            if batch_index % args.log_every == 0:
                if trainable == trainables[0]:
                    log.info('Batch:{}/{} '.format(batch_index, loader_size))
                batch_loss_str = '{:.4f}/{:.4f} ({:.4f}/{:.4f})'.format(
                    trainable.get_last_batch_loss() / args.batch_size,
                    trainable.get_last_batch_epe() / args.batch_size,
                    trainable.get_average_batch_loss() / args.batch_size,
                    trainable.get_average_batch_epe() / args.batch_size)
                log.info('\tLoss/EPE:{}\tModel:{}{}'.format(
                    batch_loss_str, trainable.get_model_name(), trainable.get_epoch()))


def test(loader, trainable, log):
    return


def main():
    scene_flow_filenames = loaders.SceneFlowFilenames.get_filenames_for_sceneflow_driving(args.scene_flow_dir)
    scene_flow_filenames.extend(loaders.SceneFlowFilenames.get_filenames_for_sceneflow_monkaa(args.scene_flow_dir))
    scene_flow_filenames.extend(loaders.SceneFlowFilenames.get_filenames_for_sceneflow_flying(args.scene_flow_dir))

    # Fix for OOM error due to copy on write of string array shared between workers.
    # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-612396143
    # manager = Manager()
    # shared_scene_flow_filenames = manager.list(scene_flow_filenames)

    image_pair_loader = torch.utils.data.DataLoader(
        # SceneFlowLoader(args.scene_flow_dir, scene_flow_filenames, size_multiple_of=81, log=log),
        SceneFlowLoader(args.scene_flow_dir, scene_flow_filenames, size_multiple_of=64, log=log),
        batch_size=args.batch_size, num_workers=12,
        pin_memory=True, drop_last=False, shuffle=True
    )

    trainables = [
        # Trainable(ElParoFlowNet(), ElParoFlowNet.loss, log, data_parallel=True),
        # Trainable(ElParoFlowNetCvA(), log, cuda_device=1), #data_parallel=True)
        # Trainable(ElParoFlowNetCvB(), log, cuda_device=1),
        # Trainable(ElParoFlowNetCvC(), log, cuda_device=0),
        # Trainable(ElParoFlowNetCvEa(), log, cuda_device=0),
        # Trainable(ElParoFlowNetCvEb(), log, cuda_device=0),
        # Trainable(ElParoFlowNetCvEc(), log, cuda_device=1),
        # Trainable(ElParoFlowNetCvEd(), log, cuda_device=1)
        # Trainable(ElParoFlowNetCvEcC(), log, cuda_device=0),
        # Trainable(ElParoFlowNetCvEcD(), log, cuda_device=1),
        # Trainable(FlowNetC(), log, cuda_device=0),
        # Trainable(FlowNetS(), log, cuda_device=1),
        # Trainable(FlowNetPwcLike(), FlowNetPwcLike.loss, log, data_parallel=True) #, cuda_device=0),
        # Trainable(FlowNetPwcLite(), FlowNetPwcLite.loss, log, data_parallel=True)
        # Trainable(FlowNetPws(), FlowNetPws.loss, log, cuda_device=0),
        # Trainable(FlowNetPws(channels=[16,64,512]), FlowNetPws.loss, log, cuda_device=1, name_postfix="S"),
        # Trainable(FlowNetPws(channels=[16, 32, 256]), FlowNetPws.loss, log, cuda_device=1, name_postfix="SS"),
        # Trainable(FlowNetPwsDeep(), FlowNetPwsDeep.loss, log, cuda_device=0),
        # Trainable(FlowNetPwsDeep(channels=[16, 64, 512]), FlowNetPwsDeep.loss, log, cuda_device=1, name_postfix="S"),
        # Trainable(FlowNetPwsDeep(channels=[16, 32, 256]), FlowNetPwsDeep.loss, log, cuda_device=1, name_postfix="SS"),
        # Trainable(FlowNetPwcLite2xA(), FlowNetPwcLite2xA.loss, log, data_parallel=True)
        Trainable(FlowNetPwc3x(), FlowNetPwc3x.loss, log, data_parallel=True)

        # Trainable(ElParoFlowNetS(), log, cuda_device=0)
        # Trainable(ElParoFlowNetM(), log, cuda_device=0),
        # Trainable(ElParoFlowNetL(), log, cuda_device=1)
    ]

    for trainable in trainables:
        trainable.load(args.save_path)

    if args.test is True:
        if trainables[0].get_epoch() == 1:
            log.error('No trained model. Give --save_path with valid checkpoint file(s)')
            return
        test(image_pair_loader, trainables[0], log)
    else:
        for epoch in range(1, args.epochs + 1):
            log.info('Starting epoch {} of {}'.format(epoch, args.epochs))
            if train(image_pair_loader, trainables):
                break

            for trainable in trainables:
                trainable.save(args.save_path)


if __name__ == '__main__':
    main()