import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from model.BTSNet import BTSNet
from data import get_loader
from utils import clip_gradient, adjust_lr
from torch.utils.tensorboard import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt,save_path
import torch.nn as nn

#train function
def train(train_loader, model, optimizer, epoch,save_path):

    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            depths=depths.cuda()

            s, s_r, s_d = model(images,depths)
            loss1 = F.binary_cross_entropy_with_logits(s, gts)
            loss2 = F.binary_cross_entropy_with_logits(s_r, gts)
            loss3 = F.binary_cross_entropy_with_logits(s_d, gts)
            loss = loss1 +loss2/2+loss3/2
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 100 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} '.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} '.
                    format( epoch, opt.epoch, i, total_step, loss1.data))
                writer.add_scalar('Loss', loss1.data, global_step=step)
                writer.add_scalar('Loss_r', loss2.data, global_step=step)
                writer.add_scalar('Loss_d', loss3.data, global_step=step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s', torch.tensor(res), step, dataformats='HW')
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                res = s_r[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s_r', torch.tensor(res), step, dataformats='HW')
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                res = s_d[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s_d', torch.tensor(res), step, dataformats='HW')
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)

                writer.add_image('Ground_truth', grid_image, step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(depths[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('depth', grid_image, step)

        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path+'/epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'/epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
        
 
if __name__ == '__main__':

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    # build the model
    model = BTSNet(nInputChannels=3, n_classes=1, os=16)
    if (opt.load is not None):
        model.load_state_dict(torch.load(opt.load),strict=False)
        print('load model from ', opt.load)
    model = nn.DataParallel(model)

    model.cuda()
    params = model.parameters()
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(num_params)
    #optimizer = torch.optim.Adadelta(filter(lambda p:p.requires_grad,model.parameters()), 0.01,weight_decay=0.0005)#opt.lr)
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),opt.lr)

    # set the path
    image_root = opt.rgb_root
    gt_root = opt.gt_root
    depth_root = opt.depth_root
    save_path = save_path()
    # load data
    print('load data...')
    train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path + '/log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Train")
    logging.info("Config")
    logging.info(
        'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
            opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
            opt.decay_epoch))

    # set loss function
    #CE = torch.nn.BCEWithLogitsLoss()

    step = 0
    writer = SummaryWriter(save_path + '/summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch,save_path)
        
