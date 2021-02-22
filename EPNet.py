import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
import os.path
import re

import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils
from models.ElParoNetModel import ElParoNet
from loaders.ImagePatchTransalationLoader import ImagePatchTranslationLoader

argParser = argparse.ArgumentParser(description="El Paro Net...")
argParser.add_argument('--epochs', type=int, default=200, help='number of epochs do')
argParser.add_argument('--train_batch_size', type=int, default=32, help='training batch size')
argParser.add_argument('--log_every', type=int, default=1000, help='Log losses after every n mini batches')
argParser.add_argument('--train_image_dir', type=str, required=True, help='dir with training images in')
argParser.add_argument('--save_path', type=str, default='checkpoints', help='dir to save model during training.')
argParser.add_argument('--patch_size', type=int, default=33, help='Size of image patches to learn')
argParser.add_argument('--max_offset', type=int, default=8, help='Usually < patch_size/4')
argParser.add_argument('--patch_scale', type=int, default=4, help='Scaling up of image patches for sampling')
argParser.add_argument('--local_cost_radius', type=int, default='9')
args = argParser.parse_args()

log = utils.logger("EPNet")

def train(loader, model, optimizer, log):
    input_patch_size = args.patch_size
    batch_loss_monitor = utils.LossMonitor()
    batch_loss_monitor_gt = utils.LossMonitor()
    loader_size = len(loader)

    patch_cost_margin = args.local_cost_size // 2
    patch_size = args.patch_size * args.patch_scale + patch_cost_margin * 2
    max_offset = args.max_offset * args.patch_scale

    axy1 = max_offset + patch_cost_margin
    axy2 = patch_size - axy1;

    model.train()

    for batch_index, (sample_patch_a, sample_patch_b, y) in enumerate(loader):
        sample_patch_a = sample_patch_a.float().cuda()
        sample_patch_b = sample_patch_b.float().cuda()
        y = y.cuda()

        optimizer.zero_grad()

        input_patch_a = sample_patch_a[:, :, patch_cost_margin:-patch_cost_margin, patch_cost_margin:-patch_cost_margin]
        input_patch_b = sample_patch_b[:, :, patch_cost_margin:-patch_cost_margin, patch_cost_margin:-patch_cost_margin]

        input_patch_a = F.interpolate(input_patch_a, size=[input_patch_size, input_patch_size], mode='bilinear', align_corners=True)
        input_patch_b = F.interpolate(input_patch_b, size=[input_patch_size, input_patch_size], mode='bilinear', align_corners=True)

        predictions = model(input_patch_a, input_patch_b)

        # loss...  from gradients of patches diffs
        better_predictions = torch.zeros_like(y)
        with torch.no_grad():
            cropped_patch_a = sample_patch_a[:, :, axy1:axy2, axy1:axy2]
            offset_predictions = predictions[:,0:2].clamp(-1,1) * max_offset

            for i in range(0, predictions.size()[0]):
                opx, opy = offset_predictions[i]
                cost, mini, maxi = utils.localCost(cropped_patch_a[i], sample_patch_b[i], opx.item(), opy.item(),
                                    local_size=args.local_cost_size)
                better = torch.as_tensor(mini, dtype=torch.float).cuda()
                better -= args.local_cost_size // 2
                better += torch.floor(offset_predictions[i])
                better /= max_offset * args.patch_scale
                better_predictions[i][0] = better[0]
                better_predictions[i][1] = better[1]

        losses = predictions - better_predictions
        losses_abs = losses.abs() # losses * losses
        losses_batch = losses_abs.sum(dim=0)
        loss = losses_batch[0] + losses_batch[1] # + losses_batch[2] / 4 + losses_batch[3] / 4
        loss.backward()
        optimizer.step()

        batch_loss_monitor.update(loss)

        losses_gt = predictions - y
        losses_gt_abs = losses_gt.abs()  # losses * losses
        losses_gt_batch = losses_gt_abs.sum(dim=0)

        batch_loss_monitor_gt.update(losses_gt_batch[0] + losses_gt_batch[1])

        if batch_index % args.log_every == 0:
            loss_str = '{:.4f} ({:.4f})'.format(
                batch_loss_monitor.last / args.train_batch_size,
                batch_loss_monitor.average() / args.train_batch_size)
            ls = [round(p, 4) for p in losses[0].tolist()]
            ys = [round(y, 4) for y in better_predictions[0].tolist()]
            log.info('[{}/{}]\t{} losses[0] Local:{} better:{}'.format(
                batch_index, loader_size, loss_str, ls, ys))

            loss_str = '{:.4f} ({:.4f})'.format(
                batch_loss_monitor_gt.last / args.train_batch_size,
                batch_loss_monitor_gt.average() / args.train_batch_size)
            ls = [round(p, 4) for p in losses_gt[0].tolist()]
            ys = [round(y, 4) for y in y[0].tolist()]
            log.info('[{}/{}]\t{} losses[0] GT:{} y:{}'.format(
                batch_index, loader_size, loss_str, ls, ys))

            # if losses_abs[0, 0] > 0.2 or losses_abs[0, 1] > 0.2:
            #     fig, ax = plt.subplots(nrows=1, ncols=2)
            #     ax.flat[0].imshow(patch_a[0].cpu(), cmap='gray')
            #     ax.flat[1].imshow(patch_b[0].cpu(), cmap='gray')
            #     plt.show()

    log.info('Epoch end average train loss GT = {}'.format(batch_loss_monitor_gt.average() / args.train_batch_size))
    log.info('Epoch end average train loss Local = {}'.format(batch_loss_monitor.average() / args.train_batch_size))

def test_accent(loader, model, log):
    in_patch_size = args.patch_size

    sample_patch_size = in_patch_size * args.patch_scale
    sample_max_offset = args.max_offset * args.patch_scale

    axy1 = sample_patch_size // 2 - sample_max_offset
    axy2 = sample_patch_size // 2 + sample_max_offset;

    for batch_index, (sample_patch_a, sample_patch_b, gt) in enumerate(loader):
        in_patch_a = F.interpolate(sample_patch_a, size=[in_patch_size, in_patch_size], mode='bilinear', align_corners=True)
        in_patch_b = F.interpolate(sample_patch_b, size=[in_patch_size, in_patch_size], mode='bilinear', align_corners=True)

        predictions = model(in_patch_a, in_patch_b)

        prediction0_xy = predictions[0,0:2].clamp(-1,1) * sample_max_offset

        log.info('GT0: {}'.format(gt[0] * sample_max_offset))
        # loss...  from gradients of patches diffs
        with torch.no_grad():
            cropped_patch_a = sample_patch_a[0, :, axy1:axy2, axy1:axy2]
            log.info('Net Prediction: {}'.format(prediction0_xy))

            for step in range(0, 10):
                opx, opy = prediction0_xy
                cost, mini, maxi = utils.localCost(cropped_patch_a, sample_patch_b[0], opx.item(), opy.item(),
                                                   local_radius=args.local_cost_radius)
                prediction0_xy[0] = mini[0]
                prediction0_xy[1] = mini[1]
                log.info('Step {}: {}'.format(step, prediction0_xy))
                if torch.equal(prediction0_xy.cpu(), gt[0] * sample_max_offset):
                    break;


            fig = plt.figure(figsize=(24, 8), dpi=112)
            ax1 = fig.add_subplot(131)
            ax1.imshow(sample_patch_a[0].squeeze(0).cpu(), cmap='gray')
            ax2 = fig.add_subplot(132)
            ax2.imshow(sample_patch_b[0].squeeze(0).cpu(), cmap='gray')
            ax3 = fig.add_subplot(133)
            ax3.imshow(cost.cpu(), cmap='gray')
            plt.show()


def main():
    max_offset = args.max_offset * args.patch_scale
    sample_patch_size = args.patch_size * args.patch_scale

    image_filenames = utils.getImageFilenamesWithPaths(args.train_image_dir)

    training_loader = torch.utils.data.DataLoader(
        ImagePatchTranslationLoader(image_filenames, patches_per_image=(args.train_batch_size * 8),
                                    patch_size=sample_patch_size, max_offset=max_offset, log=log),
        batch_size=args.train_batch_size, num_workers=0, #args.train_batch_size,
        pin_memory=False, drop_last=False
    )

    model = ElParoNet(args.patch_size)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters())
    parameter_count = sum([p.data.nelement() for p in model.parameters()])
    log.info('Model parameter count: {}'.format(parameter_count))

    start_epoch = 1

    last_saved_checkpoint = -1
    if os.path.isdir(args.save_path):
        for filename in os.listdir(args.save_path):
            match = re.match('cp(\d+).pth', filename)
            if match and last_saved_checkpoint < int(match.group(1)):
                last_saved_checkpoint = int(match.group(1))
    else:
        os.mkdir(args.save_path)

    if last_saved_checkpoint >= 0:
        model_save_path = args.save_path + '/cp' + str(last_saved_checkpoint) + '.pth'
        checkpoint = torch.load(model_save_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.info("=> loaded checkpoint {}".format(last_saved_checkpoint))

    end_epoch = start_epoch + args.epochs

    test_accent(training_loader, model, log)
    return -1

    for epoch in range(start_epoch, end_epoch):
        log.info('Starting epoch {} in range {} to {}'.format(epoch, start_epoch, end_epoch))
        train(training_loader, model, optimizer, log)

        model_save_path = args.save_path + '/cp' + str(epoch) + '.pth'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            model_save_path
            )
        log.info('Saved model for epoch {} to {}'.format(epoch, model_save_path))

if __name__ == '__main__':
    main()
