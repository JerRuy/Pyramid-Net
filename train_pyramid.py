from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import time
import numpy as np
import torch
from torch.autograd import Variable as V

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from networks.pyramid import CE_Net_
from networks.pyramid import PUNet
from scripts.loss import  *
from scripts.data import ImageFolder
from scripts.quant import *
os.environ['CUDA_VISIBLE_DEVICES'] = "2"







def QME_Net_Train(net,flagQ, parameter_mask, old_weight_dict,  image_path,  save_path,  alpha, epoch, batchsize, lr, NUM_UPDATE_LR, INITAL_EPOCH_LOSS):

    print(epoch)
    NAME = 'QME-Net_' + image_path.split('/')[-1]
    no_optim = 0
    total_epoch = epoch
    train_epoch_best_loss = INITAL_EPOCH_LOSS
    batchsize = batchsize

    dataset = ImageFolder(root_path=image_path, datasets='DRIVE')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss = dice_bce_loss()

    tic = time()
    for epoch in range(1, total_epoch + 1):

        train_epoch_loss = 0

        data_loader_iter = iter(data_loader)

        for img, mask in data_loader_iter:
            img = V(img.cuda(), volatile=False)
            mask = V(mask.cuda(), volatile=False)
            optimizer.zero_grad()
            pred = net.forward(img)

            train_loss = loss(mask, pred)


            train_loss.backward()
            optimizer.step()


            train_epoch_loss += train_loss



        train_epoch_loss = train_epoch_loss / len(data_loader_iter)


        print(' epoch: ', epoch, ' time:', int(time() - tic), ' loss1: ', round(train_epoch_loss.item(), 6))


        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            torch.save(net.state_dict(), save_path + NAME + '.th')
        #
        if no_optim > NUM_UPDATE_LR:
            net.load_state_dict(torch.load(save_path + NAME + '.th'))
            lr = lr / 2

    print('Finish!')
    return net























if __name__ == '__main__':
    print(torch.__version__)


    parser = ArgumentParser(description="Training script for QME-Net models",formatter_class=ArgumentDefaultsHelpFormatter)


    parser.add_argument('--backbone', default='cenet', type=str, help='Neural network architecture')
    parser.add_argument('--input-size', default=448, type=int, help='Images input size')
    parser.add_argument('--image-path', default='./dataset/DRIVE', type=str, help='DIRVE dataset path')
    parser.add_argument('--save-path', default='weights/', type=str, help='store the trained model')
    parser.add_argument('--alpha', default=0.005, type=int, help='the empirical coefficient for diversity')
    parser.add_argument('--epoch', default=300, type=int, help='Training epochs for full-precision stage')
    parser.add_argument('--epochQ', default=300, type=int, help='Training epochs for quantization stage')
    parser.add_argument('--batchsize', default=4, type=int, help='Batch per GPU')
    parser.add_argument('--lr', default=2e-4, type=int, help='learning rate')
    parser.add_argument('--update-lr', default=10, type=int, help='learning rate decay')
    parser.add_argument('--epochloss-init', default=10000, type=int, help='learning rate decay')
    parser.add_argument('--Q0', default=16, type=int, help='Quantization bits for meta learner B0')
    parser.add_argument('--Q1', default=4, type=int, help='Quantization bits for base learner B1')
    parser.add_argument('--Q2', default=5, type=int, help='Quantization bits for base learner B2')
    parser.add_argument('--Q3', default=6, type=int, help='Quantization bits for base learner B3')
    parser.add_argument('--quant', default=[0.3,0.3,0.2,0.2], nargs = '+', type = int, help=' ')

    args = parser.parse_args()

    # net = CE_Net_().cuda()
    net = PUNet().cuda()


    print('--------------------------------------------------------------')
    print('--------------------Full-precision stage----------------------')
    print('--------------------------------------------------------------')

    flagQ = False
    net = QME_Net_Train(net, flagQ, [], [],   args.image_path,  args.save_path, args.alpha, args.epoch, args.batchsize, args.lr,args.update_lr, args.epochloss_init )

    # model_path = './weights/QME-Net_DRIVE.th'
    # net = loadweight(net, model_path)














