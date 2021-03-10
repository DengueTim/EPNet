import argparse
import torch
import torch.utils.data
from multiprocessing import Manager

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as d
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
from models.ElParoFlowNetModelCvA import ElParoFlowNetCvA
from models.Trainable import Trainable

argParser = argparse.ArgumentParser(description="El Paro Net...")
argParser.add_argument('--epochs', type=int, default=100, help='number of epochs do')
argParser.add_argument('--batch_size', type=int, default=32, help='training batch size')
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
        (batch_size, image_c, image_h, image_w) = image_a.size()
        # image_a = image_a.cuda()
        # image_b = image_b.cuda()
        # flow_a2b_gt = flow_a2b_gt.cuda()
        #flow_b2a_gt = flow_b2a_gt.cuda()

        scaled_h = image_h // scale
        scaled_w = image_w // scale

        image_a_x15 = F.interpolate(image_a, size=(scaled_h, scaled_w), mode='bilinear', align_corners=True)
        image_b_x15 = F.interpolate(image_b, size=(scaled_h, scaled_w), mode='bilinear', align_corners=True)

        # For ElParoFlowNet: boarder = 9 * 15
        boarder = 2
        flow_a2b_gt_x15 = F.interpolate(flow_a2b_gt, size=(scaled_h, scaled_w), mode='area')
        flow_a2b_gt_cropped_x15 = flow_a2b_gt_x15[:, :, boarder:-boarder, boarder:-boarder]

        for trainable in trainables:
            trainable.zero_grad()

            flow_prediction_x15 = trainable(image_a_x15, image_b_x15)
           # flow_prediction = F.interpolate(flow_prediction_x15, scale_factor=scale, mode='bilinear', align_corners=True)

            # fig = plt.figure(figsize=(18, 12), dpi=112)
            # ax1 = fig.add_subplot(221)
            # ax1.imshow(image_a[0].squeeze(0).cpu(), cmap='gray')
            # ax2 = fig.add_subplot(222)
            # ax2.imshow(image_b[0].squeeze(0).cpu(), cmap='gray')
            # ax3 = fig.add_subplot(223)
            # # ax3.hist(torch.flatten(flow_a2b_gt_cropped_x15[0][0].detach().cpu()).numpy(), bins=100)
            # ax3.imshow(color_flow.flow_to_rgb(flow_a2b_gt_cropped_x15[0].detach().cpu().squeeze(0), rad_normaliser=100))
            # ax4 = fig.add_subplot(224)
            # # ax4.hist(torch.flatten(flow_a2b_gt_cropped_x15[0][1].detach().cpu()).numpy(), bins=100)
            # ax4.imshow(color_flow.flow_to_rgb(flow_prediction_x15[0].detach().cpu().squeeze(0), rad_normaliser=100))
            # plt.show()

            mask = torch.ones_like(flow_a2b_gt_cropped_x15, dtype=bool)
            count = mask[:, 0, :, :].sum(dim=2).sum(dim=1)
            difference = flow_a2b_gt_cropped_x15 - flow_prediction_x15
            per_pixel_losses = torch.sqrt(torch.sum(torch.pow(difference, 2), dim=1))
            mask_size = mask.size()
            loss = torch.sum(per_pixel_losses) / (mask_size[0] * mask_size[2] * mask_size[3])
            loss.backward()
            trainable.step()
            trainable.update_batch_loss(loss)

            if batch_index % args.log_every == 0:
                if trainable == trainables[0]:
                    log.info('Batch:{}/{} '.format(batch_index, loader_size))
                batch_loss_str = '{:.4f} ({:.4f})'.format(
                    trainable.get_last_batch_loss() / args.batch_size,
                    trainable.get_average_batch_loss() / args.batch_size)
                log.info('\tLoss:{}\tModel:{}{}'.format(
                    batch_loss_str, trainable.get_model_name(), trainable.get_epoch()))


def test(loader, trainable, log):
    return


def main():
    scene_flow_filenames = loaders.SceneFlowFilenames.get_filenames_for_sceneflow_driving(args.scene_flow_dir + '/Driving')
    scene_flow_filenames.extend(loaders.SceneFlowFilenames.get_filenames_for_sceneflow_monkaa(args.scene_flow_dir + '/Monkaa'))

    # Fix for OOM error due to copy on write of string array shared between workers.
    # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-612396143
    # manager = Manager()
    # shared_scene_flow_filenames = manager.list(scene_flow_filenames)

    image_pair_loader = torch.utils.data.DataLoader(
        SceneFlowLoader(scene_flow_filenames, log=log),
        batch_size=args.batch_size, num_workers=8,
        pin_memory=True, drop_last=False, shuffle=True
    )

    trainables = [
        Trainable(ElParoFlowNet(), log, cuda_device=0), #data_parallel=True),
        Trainable(ElParoFlowNetCvA(), log, cuda_device=1), #data_parallel=True)
        Trainable(ElParoFlowNetCvB(), log, cuda_device=1)
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
            train(image_pair_loader, trainables)

            for trainable in trainables:
                trainable.save(args.save_path)


if __name__ == '__main__':
    main()