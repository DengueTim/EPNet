import torch
import torch.nn as nn
import torch.optim as optim

import os
import os.path
import re


class Trainable():
    def __init__(self, model, log, lr=1e-3, data_parallel=False, cuda_device=None):
        self.model_name = model.__class__.__name__

        if cuda_device is not None:
            model = model.cuda(cuda_device)
        elif data_parallel:
            model = nn.parallel.DataParallel(model).cuda()
            #model = nn.parallel.DistributedDataParallel(model)

        self.data_parallel = data_parallel
        self.model = model
        self.cuda_device = cuda_device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.epoch = 0

        parameter_count = sum([p.data.nelement() for p in model.parameters()])
        log.info('Parameter count {} for model {}'.format(parameter_count, self.model_name))

        self.log = log

        self.last_batch_loss = 0
        self.batch_loss_sum = 0
        self.batch_counter = 0

    def get_model_name(self):
        return self.model_name

    def train(self):
        self.model.train()
        self.epoch += 1
        self.last_batch_loss = 0
        self.batch_loss_sum = 0
        self.batch_counter = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def __call__(self, *args, **kwargs):
        if self.data_parallel is True:
            return self.model(*args, **kwargs).cpu()

        if self.cuda_device is None:
            return self.model(*args, **kwargs)

        a = []
        for arg in args:
            a.extend([arg.cuda(self.cuda_device)])

        return self.model(*a, **kwargs).cpu()

    def step(self):
        self.optimizer.step()

    def get_epoch(self):
        return self.epoch

    def load(self, save_path):
        last_saved_checkpoint = -1
        if os.path.isdir(save_path):
            file_name_pattern = 'cp_{}(\d+).pth'.format(self.get_model_name())
            file_name_matcher = re.compile(file_name_pattern)
            for filename in os.listdir(save_path):
                match = file_name_matcher.match(filename)
                if match and last_saved_checkpoint < int(match.group(1)):
                    last_saved_checkpoint = int(match.group(1))
        else:
            os.mkdir(save_path)

        if last_saved_checkpoint == -1:
            return 0

        model_save_path = save_path + '/cp_' + self.get_model_name() + str(last_saved_checkpoint) + '.pth'
        checkpoint = torch.load(model_save_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.log.info("=> loaded checkpoint from {}".format(model_save_path))
        return self.epoch

    def save(self, save_path):
        model_save_path = save_path + '/cp_' + self.get_model_name() + str(self.epoch) + '.pth'
        torch.save({
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()},
            model_save_path
        )
        self.log.info('Saved model for epoch {} to {}'.format(self.epoch, model_save_path))

    def get_last_batch_loss(self):
        return self.last_batch_loss

    def update_batch_loss(self, loss):
        self.last_batch_loss = loss
        self.batch_loss_sum += loss
        self.batch_counter += 1

    def get_average_batch_loss(self):
        return self.batch_loss_sum / self.batch_counter if self.batch_counter else 0
