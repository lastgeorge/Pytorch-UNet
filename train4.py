#!/usr/bin/env python

import sys
import os
import math
import itertools
import json
from datetime import datetime
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet
from mobilenetv2 import MobileNet_UNet
from mobilenetv3 import MobileNetV3_UNet
from transformer import Transformer_UNet

from eval_util import eval_dice, eval_loss, eval_eff_pur
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, chw_to_hwc
from utils import h5_utils as h5u

def build_model(model_name, n_channels, n_classes, mobilenetv3_variant="large", pretrained=True):
    model_name = model_name.lower()

    if model_name == "unet":
        return UNet(n_channels, n_classes)

    elif model_name == "uresnet":
        return UResNet(n_channels, n_classes)

    elif model_name == "nestedunet":
        return NestedUNet(n_channels, n_classes)

    elif model_name == "mobilenetv2":
        return MobileNet_UNet(n_channels, n_classes)

    elif model_name == "mobilenetv3":
        return MobileNetV3_UNet(
            n_channels,
            n_classes,
            variant=mobilenetv3_variant,
            pretrained=pretrained
        )

    elif model_name == "transformer":
        return Transformer_UNet(
            n_channels,
            n_classes,
            embed_dim=512,
            num_heads=8,
            depth=4
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

def lr_exp_decay(optimizer, lr0, gamma, epoch):
    lr = lr0*math.exp(-gamma*epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train_net(net,
              device,
              im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0'],
              # ma_tags = ['frame_ductor0'],
              ma_tags = ['frame_deposplat0'],
              rebin_factor = 10,
              truth_th = 100,
              file_img  = [f"/nfs/data/1/hnam/train_data_PDHD_fixedbug_separateWC/modified/g4-rec-{i}_modified.h5" for i in range(60) if i!=2],
              file_mask = [f"/nfs/data/1/hnam/train_data_PDHD_fixedbug_separateWC/modified/g4-tru-{i}_modified.h5" for i in range(60) if i!=2],
              sepoch=0,
              nepoch=1,
              strain=0,
              ntrain=10,
              sval=450,
              nval=50, 
              batch_size=10,
              lr=0.1,
              val_percent=0.10,
              save_cp=True,
              gpu=False,
              img_scale=0.5,
              padding=0,
              min_run=1,
              padding_side='both',
              avoid_merge=True,
              min_gap=1,
              model_name="mobilenetv3"):

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_checkpoint = f"chk_{model_name}_{run_id}/"

    os.makedirs(dir_checkpoint, exist_ok=True)
    with open(os.path.join(dir_checkpoint, "config.json"), "w") as f:
        json.dump({
            "model": model_name,
            "rebin": rebin_factor,
            "truth_th": truth_th,
            "padding": padding,
            "min_run": min_run,
            "padding_side": padding_side,
            "avoid_merge": avoid_merge,
            "min_gap": min_gap,
            "nepoch": nepoch,
            "learning_rate": lr,
            "batch_size": batch_size,
            "val_percent": nval/(sval + nval),
        }, f, indent=2)

    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    iddataset = {}
    event_per_file = 10
    event_zero_id_offset_rec = 100
    event_zero_id_offset_tru = 0
    
    def id_gen_rec(index):
        return (index // event_per_file, index % event_per_file + event_zero_id_offset_rec)
    def id_gen_tru(index):
        return (index // event_per_file, index % event_per_file + event_zero_id_offset_tru)
    
    iddataset['train_rec'] = [id_gen_rec(i) for i in list(strain+np.arange(ntrain))]
    iddataset['train_tru'] = [id_gen_tru(i) for i in list(strain+np.arange(ntrain))]
    iddataset['val_rec'] = [id_gen_rec(i) for i in list(sval+np.arange(nval))]
    iddataset['val_tru'] = [id_gen_tru(i) for i in list(sval+np.arange(nval))]

    outfile_log = open(dir_checkpoint+'/log','a+')

    print("Train REC IDs:", iddataset['train_rec'], file=outfile_log, flush=True)
    print("Train TRU IDs:", iddataset['train_tru'], file=outfile_log, flush=True)
    print("Validation REC IDs:", iddataset['val_rec'], file=outfile_log, flush=True)
    print("Validation TRU IDs:", iddataset['val_tru'], file=outfile_log, flush=True)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(nepoch, batch_size, lr, len(iddataset['train_rec']),
               len(iddataset['val_rec']), str(save_cp), str(gpu)), file=outfile_log, flush=True)

    N_train = len(iddataset['train_rec'])

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    
    criterion = nn.BCELoss()

    print(f"""
    im_tags: {im_tags}
    ma_tags: {ma_tags}
    truth_th: {truth_th}
    padding: {padding}, min_run: {min_run}, padding_side: {padding_side},
    avoid_merge: {avoid_merge}, min_gap: {min_gap}
    """, file=outfile_log, flush=True)

    outfile_loss_batch = open(dir_checkpoint+'/loss-batch.csv','a+')
    outfile_loss       = open(dir_checkpoint+'/loss.csv','a+')
    outfile_eval_dice  = open(dir_checkpoint+'/eval-dice.csv','a+')
    outfile_eval_loss  = open(dir_checkpoint+'/eval-loss.csv','a+')

    eval_labels = [
        '75-75',
        '87-85',
    ]
    eval_imgs = []
    eval_masks = []
    for label in eval_labels:
        eval_imgs.append('eval/eval-'+label+'/g4-rec-0.h5')
        eval_masks.append('eval/eval-'+label+'/g4-tru-0.h5') 
    outfile_ep = []
    for label in eval_labels:
        outfile_ep.append(open(dir_checkpoint+'/ep-'+label+'.csv','a+'))
    
    if sepoch > 0 :
        net.load_state_dict(torch.load('{}/CP{}.pth'.format(dir_checkpoint, sepoch-1)))
    
    for epoch in range(sepoch,sepoch+nepoch):
        # scheduler = lr_exp_decay(optimizer, lr, 0.04, epoch)
        scheduler = optimizer
        
        print('epoch: {} start'.format(epoch))
        print(optimizer, file=outfile_log, flush=True)
        
        y_range_dict = {
                1: 6000,
                2: 3000,
                3: 2000,
                4: 1500,
                5: 1200,
                6: 1000,
                8: 750,
                10: 600
        }
        rebin = [1, rebin_factor]
        y_range = [0, y_range_dict.get(rebin_factor, 600)]

        #x_range = [0, 800] # PDHD, U, left-closed right-open interval
        #x_range = [800, 1600] # PDHD, V, left-closed right-open interval
        x_range = [0, 1600] # PDHD, Induction, left-closed right-open interval
        # x_range = [476, 952] # PDVD, V
        z_scale = 4000
        
        print('''
        file_img: {}
        file_mask: {}
        '''.format(file_img, file_mask), file=outfile_log, flush=True)

        print('Starting epoch {}/{}.'.format(epoch, nepoch))
        net.train()

        train = zip(
          h5u.get_chw_imgs(file_img, iddataset['train_rec'], im_tags, rebin, x_range, y_range, z_scale),
          h5u.get_masks(file_mask,   iddataset['train_tru'], ma_tags, rebin, x_range, y_range, truth_th, padding, min_run, padding_side, avoid_merge, min_gap)
        )
        val = zip(
          h5u.get_chw_imgs(file_img, iddataset['val_rec'],   im_tags, rebin, x_range, y_range, z_scale),
          h5u.get_masks(file_mask,   iddataset['val_tru'],   ma_tags, rebin, x_range, y_range, truth_th, padding, min_run, padding_side, avoid_merge, min_gap)
        )
        eval_data = []
        for i in range(len(eval_imgs)):
            id_eval = [0]
            eval_data.append(
                zip(
                    h5u.get_chw_imgs(eval_imgs[i], id_eval,   im_tags, rebin, x_range, y_range, z_scale),
                    h5u.get_masks(eval_masks[i],   id_eval,   ma_tags, rebin, x_range, y_range, truth_th, padding, min_run, padding_side, avoid_merge, min_gap)
                )
            )

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])
            if False:
                h5u.plot_mask(b[0][1])
                h5u.plot_img(chw_to_hwc(b[0][0]))

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.to(device)
                # print(f">>> imgs.shape: {imgs.shape}")
                true_masks = true_masks.to(device)
                # imgs = imgs.cuda()
                # true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{} : {:.4f} --- loss: {:.6f}'.format(epoch, i * batch_size / N_train, loss.item()))
            print('{:.4f}, {:.6f}'.format(i * batch_size / N_train, loss.item()), file=outfile_loss_batch, flush=True)
            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            scheduler.step()

        epoch_loss = epoch_loss / (i + 1)
        print('Epoch finished ! Loss: {:.6f}'.format(epoch_loss))
        print('{:.4f}, {:.6f}'.format(epoch, epoch_loss), file=outfile_loss, flush=True)

        if save_cp:
            torch.save(net.state_dict(),
                      dir_checkpoint + 'CP{}.pth'.format(epoch))
            print('Checkpoint e{} saved !'.format(epoch))

        if True:
            val1, val2 = itertools.tee(val, 2)
            
            # val_dice = eval_dice(net, val1, gpu)
            # print('Validation Dice Coeff: {:.4f}, {:.6f}'.format(epoch, val_dice))
            # print('{:.4f}, {:.6f}'.format(epoch, val_dice), file=outfile_eval_dice, flush=True)

            # val_loss = eval_loss(net, criterion, val2, gpu)
            # print('Validation Loss: {:.4f}, {:.6f}'.format(epoch, val_loss))
            # print('{:.4f}, {:.6f}'.format(epoch, val_loss), file=outfile_eval_loss, flush=True)
            if gpu:
                val_loss = eval_loss(net, criterion, val2, gpu, device)
            else:
                val_loss = eval_loss(net, criterion, val2, gpu)
            print('Validation Loss: {:.4f}, {:.6f}'.format(epoch, val_loss))
            print('{:.4f}, {:.6f}'.format(epoch, val_loss), file=outfile_eval_loss, flush=True)
            
            # for data, out in zip(eval_data,outfile_ep):
            #     ep = eval_eff_pur(net, data, 0.5, gpu)
            #     print('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(epoch, ep[0], ep[1], ep[2], ep[3]), file=out, flush=True)
            



def get_args():
    parser = OptionParser()

    parser.add_option('--model', dest='model', type='choice',
                      choices=['unet', 'uresnet', 'nestedunet',
                               'mobilenetv2', 'mobilenetv3', 'transformer'],
                      default='mobilenetv3',
                      help='DNN model to train: unet | uresnet | nestedunet | mobilenetv2 | mobilenetv3 | transformer')

    parser.add_option('--mobilenetv3-variant', dest='mobilenetv3_variant',
                      type='choice', choices=['small', 'large'],
                      default='large',
                      help='MobileNetV3 variant: small | large')

    parser.add_option('--no-pretrained', action='store_false',
                      dest='pretrained', default=True,
                      help='disable pretrained backbone for MobileNetV3')

    parser.add_option('--start-epoch', dest='sepoch', default=0, type='int',
                      help='start epoch number')
    parser.add_option('-e', '--nepoch', dest='nepoch', default=1, type='int',
                      help='number of epochs')

    parser.add_option('--start-train', dest='strain', default=0, type='int',
                      help='start sample for training')
    parser.add_option('--ntrain', dest='ntrain', default=10, type='int',
                      help='number of sample for training')
    parser.add_option('--start-val', dest='sval', default=450, type='int',
                      help='start sample for val')
    parser.add_option('--nval', dest='nval', default=50, type='int',
                      help='number of sample for nval')

    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('--gpu-id', dest='gpu_id', default=0, type='int',
                      help='which GPU to use (default: 0)')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('--rebin', dest='rebin', type='int',
                  default=10, help='downsampling factor for wire/time axis (e.g., 1, 4, 6, 10)')
    parser.add_option('-t', '--truth-th', dest='truth_th', type='int',
                      default=100, help='threshold for truth mask binarization')
    parser.add_option('-p', '--padding', dest='padding', type='int',
                      default=0, help='time-axis padding added to both ends of signals')
    parser.add_option('--min-run', dest='min_run', type='int',
                      default=1, help='minimum consecutive truth-length (in time) for padding')
    parser.add_option('--padding-side', dest='padding_side', type='string',
                      default='both', help="padding side: 'both' | 'left' | 'right'")
    parser.add_option('--avoid-merge', action='store_true', dest='avoid_merge',
                      default=False, help='reduce padding to keep at least min_gap zeros between runs')
    parser.add_option('--min-gap', dest='min_gap', type='int',
                      default=1, help='minimum number of zeros to keep between runs when avoid_merge=True')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    torch.set_num_threads(1)

    # im_tags = ['frame_tight_lf0', 'frame_loose_lf0'] #lt
    im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # l23
    # im_tags = ['frame_loose_lf0', 'frame_tight_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # lt23
    # ma_tags = ['frame_ductor0']
    ma_tags = ['frame_deposplat0']

    net = build_model(
        model_name=args.model,
        n_channels=len(im_tags),
        n_classes=len(ma_tags),
        mobilenetv3_variant=args.mobilenetv3_variant,
        pretrained=args.pretrained
    )

    # net = UNet(len(im_tags), len(ma_tags))
    # net = UResNet(len(im_tags), len(ma_tags))
    # net = NestedUNet(len(im_tags),len(ma_tags))
    # net = MobileNet_UNet(len(im_tags), len(ma_tags))
    # net = MobileNetV3_UNet(len(im_tags), len(ma_tags), variant="large", pretrained=True)
    # net = Transformer_UNet(len(im_tags), len(ma_tags), embed_dim=512, num_heads=8, depth=4)

    print(f"[INFO] Model = {args.model}")

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        device = torch.device("cuda")
        net.to(device)
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        #print("CUDA on")
        #net.cuda()
        #device = torch.device(f"cuda:{args.gpu_id}")
        #print(f"CUDA on: using GPU {args.gpu_id}")
        #net.to(device)
        # cudnn.benchmark = True # faster convolutions, but more memory
    else:
        device = torch.device("cpu")

    print(f"[INFO] Using truth threshold = {args.truth_th}")
    try:
        train_net(net=net,
                  device=device,
                  im_tags=im_tags,
                  ma_tags=ma_tags,
                  rebin_factor=args.rebin,
                  truth_th=args.truth_th,
                  sepoch=args.sepoch,
                  nepoch=args.nepoch,
                  strain=args.strain,
                  ntrain=args.ntrain,
                  sval=args.sval,
                  nval=args.nval,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  padding=args.padding,
                  min_run=args.min_run,
                  padding_side=args.padding_side,
                  avoid_merge=args.avoid_merge,
                  min_gap=args.min_gap,
                  model_name=args.model)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
