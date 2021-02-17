import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
import os.path
import re

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
args = argParser.parse_args()

log = utils.logger("EPNet")

def train(loader, model, optimizer, log):
    batch_loss_monitor = utils.LossMonitor()
    loader_size = len(loader)

    model.train()

    for batch_index, (patch_a, patch_b, y) in enumerate(loader):
        patch_a = patch_a.float().cuda()
        patch_b = patch_b.float().cuda()
        y = y.cuda()

        optimizer.zero_grad()

        predictions = model(patch_a, patch_b)
        losses = predictions - y
        losses_abs = losses.abs() # losses * losses
        losses_batch = losses_abs.sum(dim=0)
        loss = losses_batch[0] + losses_batch[1] + losses_batch[2] / 4 + losses_batch[3] / 4
        loss.backward()
        optimizer.step()

        batch_loss_monitor.update(loss)

        if batch_index % args.log_every == 0:
            loss_str = '{:.4f} ({:.4f})'.format(
                batch_loss_monitor.last / args.train_batch_size,
                batch_loss_monitor.average() / args.train_batch_size)
            ls = [round(p, 4) for p in losses[0].tolist()]
            ys = [round(y, 4) for y in y[0].tolist()]
            log.info('[{}/{}]\t{} losses0:{} y0:{}'.format(
                batch_index, loader_size, loss_str, ls, ys))

            # if losses_abs[0, 0] > 0.2 or losses_abs[0, 1] > 0.2:
            #     fig, ax = plt.subplots(nrows=1, ncols=2)
            #     ax.flat[0].imshow(patch_a[0].cpu(), cmap='gray')
            #     ax.flat[1].imshow(patch_b[0].cpu(), cmap='gray')
            #     plt.show()

    log.info('Epoch end average train loss = {}'.format(batch_loss_monitor.average() / args.train_batch_size))

def main():
    image_filenames = utils.getImageFilenamesWithPaths(args.train_image_dir)

    training_loader = torch.utils.data.DataLoader(
        ImagePatchTranslationLoader(image_filenames, patches_per_image=(args.train_batch_size * 8), log=log),
        batch_size=args.train_batch_size, num_workers=args.train_batch_size,
        pin_memory=True, drop_last=False
    )

    model = ElParoNet()
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
