import argparse
import torch
import torch.utils.data
from multiprocessing import Manager

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import utils
from models.ElParoNetModelSmallest import ElParoNetSmallest
from models.ElParoNetModelSmaller import ElParoNetSmaller
from models.ElParoNetModel import ElParoNet
from models.ElParoNetModelBigger import ElParoNetBigger
from models.Trainable import Trainable
from loaders.ImagePatchTransalationLoader import ImagePatchTranslationLoader

argParser = argparse.ArgumentParser(description="El Paro Net...")
argParser.add_argument('--epochs', type=int, default=100, help='number of epochs do')
argParser.add_argument('--batch_size', type=int, default=64, help='training batch size')
argParser.add_argument('--log_every', type=int, default=1000, help='Log losses after every n mini batches')
argParser.add_argument('--image_dir', type=str, required=True, help='dir with train/test images in')
argParser.add_argument('--save_path', type=str, default='checkpoints', help='dir to save model during training.')
argParser.add_argument('--patch_size', type=int, default=33, help='Size of image patches to learn')
argParser.add_argument('--max_offset', type=int, default=8, help='Usually < patch_size/4')
argParser.add_argument('--patch_scale', type=int, default=4, help='Scaling up of image patches for sampling')
argParser.add_argument('--test', type=bool, default=False, help='Testing not training')
args = argParser.parse_args()

log = utils.logger("EPNet")


def train(loader, trainables):
    in_patch_size = args.patch_size

    loader_size = len(loader)

    # sample_patch_size = in_patch_size * args.patch_scale
    # sample_max_offset = args.max_offset * args.patch_scale

    # axy1 = sample_patch_size // 2 - sample_max_offset
    # axy2 = sample_patch_size // 2 + sample_max_offset;

    for trainable in trainables:
        trainable.train()

    for batch_index, (sample_patch_a, sample_patch_b, gt) in enumerate(loader):
        sample_patch_a = sample_patch_a.cuda()
        sample_patch_b = sample_patch_b.cuda()

        in_patch_a = F.interpolate(sample_patch_a, size=[in_patch_size, in_patch_size], mode='bilinear', align_corners=True)
        in_patch_b = F.interpolate(sample_patch_b, size=[in_patch_size, in_patch_size], mode='bilinear', align_corners=True)

        gt_xy = gt.cuda()

        for trainable in trainables:
            trainable.zero_grad()

            predictions = trainable(in_patch_a, in_patch_b)

            losses_xys = predictions[:, :2] - gt_xy
            losses_xys_2 = losses_xys * losses_xys
            losses_xys_batch = losses_xys_2.sum()

            if losses_xys_batch / args.batch_size > 0.05:
                losses_confidences = predictions[:,2:4] * 0
                losses_confidence_batch = 0 #losses_confidences.sum()
            else:
                # Try and predict the loss itself...
                losses_confidences = predictions[:, 2:4] - losses_xys.abs()
                losses_confidences_2 = losses_confidences * losses_confidences
                losses_confidence_batch = losses_confidences_2.sum()

            loss = 0.7 * losses_xys_batch + 0.3 * losses_confidence_batch
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.model.parameters(), 0.5)
            trainable.step()

            trainable.update_batch_loss(loss)

            # losses_gt = predictions - gt.cuda()
            # losses_gt_abs = losses_gt.abs()  # losses * losses
            # losses_gt_batch = losses_gt_abs.sum(dim=0)
            #
            # batch_loss_monitor_gt.update(losses_gt_batch[0] + losses_gt_batch[1])

            if batch_index % args.log_every == 0:
                if trainable == trainables[0]:
                    gt0 = [round(y, 4) for y in gt[0].tolist()]
                    log.info('Batch:{}/{} GT0:{} '.format(batch_index, loader_size, gt0))
                batch_loss_str = '{:.4f} ({:.4f})'.format(
                    trainable.get_last_batch_loss() / args.batch_size,
                    trainable.get_average_batch_loss() / args.batch_size)
                err0xy = [round(p, 4) for p in losses_xys[0].tolist()]
                pred0err = [round(p, 4) for p in predictions[0, 2:4].tolist()]
                log.info('\tLoss:{}\tSample0 Err XY:{} PredErr:{}\tModel:{}{}'.format(
                    batch_loss_str, err0xy, pred0err, trainable.get_model_name(), trainable.get_epoch()))
                #
                # loss_str = '{:.4f} ({:.4f})'.format(
                #     batch_loss_monitor_gt.last / args.batch_size,
                #     batch_loss_monitor_gt.average() / args.batch_size)
                # ls = [round(p, 4) for p in losses_gt[0].tolist()]
                # ys = [round(y, 4) for y in gt[0].tolist()]
                # log.info('[{}/{}]\t{} losses[0] GT:{} y:{}'.format(
                #     batch_index, loader_size, loss_str, ls, ys))

                # if losses_abs[0, 0] > 0.2 or losses_abs[0, 1] > 0.2:
                #     fig, ax = plt.subplots(nrows=1, ncols=2)
                #     ax.flat[0].imshow(patch_a[0].cpu(), cmap='gray')
                #     ax.flat[1].imshow(patch_b[0].cpu(), cmap='gray')
                #     plt.show()

    for trainable in trainables:
        log.info('Epoch end average train loss = {}'.format(trainable.get_average_batch_loss() / args.batch_size))


def test_accent(loader, trainable, log):
    in_patch_size = args.patch_size

    sample_patch_size = in_patch_size * args.patch_scale
    sample_max_offset = args.max_offset * args.patch_scale

    axy1 = sample_patch_size // 2 - sample_max_offset
    axy2 = sample_patch_size // 2 + sample_max_offset

    for batch_index, (sample_patch_a, sample_patch_b, gt) in enumerate(loader):
        in_patch_a = F.interpolate(sample_patch_a, size=[in_patch_size, in_patch_size], mode='bilinear', align_corners=True)
        in_patch_b = F.interpolate(sample_patch_b, size=[in_patch_size, in_patch_size], mode='bilinear', align_corners=True)

        predictions = trainable(in_patch_a, in_patch_b)

        prediction0_xy = predictions[0,0:2].clamp(-1,1) * sample_max_offset

        log.info('GT0: {}'.format(gt[0] * sample_max_offset))
        # loss...  from gradients of patches diffs
        with torch.no_grad():
            cropped_patch_a = sample_patch_a[0, :, axy1:axy2, axy1:axy2]
            log.info('Net Prediction: {}'.format(prediction0_xy))

            for step in range(0, 20):
                opx, opy = prediction0_xy
                cost, mini, maxi = utils.localCost(cropped_patch_a, sample_patch_b[0], opx.item(), opy.item(),
                                                   std_dev=args.max_offset)
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
            ax3.imshow(cost, cmap='gray')
            plt.show()


def test(loader, trainable, log):
    in_patch_size = args.patch_size

    sample_patch_size = in_patch_size * args.patch_scale
    sample_max_offset = args.max_offset * args.patch_scale

    axy1 = sample_patch_size // 2 - sample_max_offset
    axy2 = sample_patch_size // 2 + sample_max_offset

    for batch_index, (sample_patch_a, sample_patch_b, sample_gt) in enumerate(loader):
        sample_patch_a = sample_patch_a.cuda()
        sample_patch_b = sample_patch_b.cuda()

        in_patch_a = F.interpolate(sample_patch_a, size=[in_patch_size, in_patch_size], mode='bilinear', align_corners=True)
        in_patch_b = F.interpolate(sample_patch_b, size=[in_patch_size, in_patch_size], mode='bilinear', align_corners=True)

        predictions = trainable(in_patch_a, in_patch_b)

        losses_xys = predictions[:, :2] - sample_gt.cuda()
        losses_xys_2 = losses_xys * losses_xys
        losses_xys_sample = losses_xys_2.sum(dim=1)

        for i in range(0, predictions.size()[0]):
            gts = [round(y, 4) for y in sample_gt[i].tolist()]
            xys = [round(p, 4) for p in losses_xys[i].tolist()]
            ps = [round(p, 4) for p in predictions[i].tolist()]
            log.info('Loss: {:.4f} GT:{} XY loss:{} Prediction:{}'.format(losses_xys_sample[i], gts, xys, ps))

            if True or losses_xys_sample[i] > 0.05:
                cropped_patch_a = sample_patch_a[i, :, axy1:axy2, axy1:axy2]
                cost, mini, maxi = utils.cost(cropped_patch_a, sample_patch_b[i])


                fig = plt.figure(figsize=(12, 12), dpi=112)
                ax1 = fig.add_subplot(221)
                ax1.imshow(in_patch_a[i].squeeze(0).cpu(), cmap='gray')
                ax2 = fig.add_subplot(222)
                ax2.imshow(in_patch_b[i].squeeze(0).cpu(), cmap='gray')
                ax3 = fig.add_subplot(223)
                ax3.imshow(sample_patch_a[i].squeeze(0).cpu(), cmap='gray')
                ax4 = fig.add_subplot(224)
                ax4.imshow(cost, cmap='gray')
                gt = sample_gt[i] * sample_max_offset + sample_max_offset + args.patch_scale // 2
                ax4.plot(gt[0], gt[1], marker='x', color="green")
                xy = predictions[i, :2].cpu().detach() * sample_max_offset + sample_max_offset + args.patch_scale // 2
                ax4.plot(xy[0], xy[1], marker='x', color="red")
                err = predictions[i, 2:4].cpu().detach().abs() * sample_max_offset * 4
                ax4.add_patch(Ellipse((xy[0], xy[1]), width=err[0], height=err[1],
                     edgecolor='red',
                     facecolor='none',
                     linewidth=1))
                plt.show()

def main():
    max_offset = args.max_offset * args.patch_scale
    sample_patch_size = args.patch_size * args.patch_scale

    image_filenames = utils.getImageFilenamesWithPaths(args.image_dir)

    # Fix for OOM error due to copy on write of string array shared between workers.
    # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-612396143
    manager = Manager()
    shared_image_filenames = manager.list(image_filenames)

    image_loader = torch.utils.data.DataLoader(
        ImagePatchTranslationLoader(shared_image_filenames, patches_per_image=(args.batch_size * 4),
                                    patch_size=sample_patch_size, max_offset=max_offset, log=log),
        batch_size=args.batch_size, num_workers=12,
        pin_memory=True, drop_last=False
    )

    trainables = [
        Trainable(ElParoNetSmallest(args.patch_size), log, lr=0.0002),
        Trainable(ElParoNetSmaller(args.patch_size), log, lr=0.0002),
        Trainable(ElParoNet(args.patch_size), log, lr=0.0002),
        Trainable(ElParoNetBigger(args.patch_size), log, lr=0.0002)
    ]


    for trainable in trainables:
        trainable.load(args.save_path)

    # test_accent(training_loader, model, log)
    # return -1

    if args.test is True:
        if trainables[3].get_epoch() == 1:
            log.error('No trained model. Give --save_path with valid checkpoint file(s)')
            return
        test(image_loader, trainables[3], log)
    else:
        for epoch in range(1, args.epochs + 1):
            log.info('Starting epoch {} of {}'.format(epoch, args.epochs))
            train(image_loader, trainables)

            for trainable in trainables:
                trainable.save(args.save_path)


if __name__ == '__main__':
    main()
