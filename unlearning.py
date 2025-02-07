# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Author      : Wenjian Luo, Qi Zhou, Zipeng Ye, Yubo Tang
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : unlearning.py
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torchvision import datasets, transforms
from facenet_test2.facenet_test.networks.facenet import Facenet
from facenet_test2.facenet_test.utils.weights_init import weights_init
import torch.backends.cudnn as cudnn
import numpy as np
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from facenet_test2.facenet_test.utils.triplet_loss import triplet_loss
from facenet_test2.facenet_test.utils.dataloder_face import CustomDataset, collate_fc,FacenetDataset
from facenet_test2.facenet_test.utils.update_one_epoch_facenet import one_epoch_update
import argparse

def get_num_classes(annotation_path):
    # return total classes of training data
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

if __name__ == "__main__":

    #---------------------------------------------------
    # parameters
    pretrained_dir  = None
    #todo:1代表传统unlearning; 2代表BN-Cleaning; 3代表BN-Unlearning
    choice = 3
    #todo:1代表投毒；0代表干净数据集
    poi_trainflag = 1
    poi_valflag = 1
    input_shape     = [160, 160, 3]
    backbone        = 'mobile_net'     # 'mobile_net' or 'inception_resnetv1'
    saved_net_path = '/home/zq/projects/facenet_test-train/facenet_test/facenet_logs/facetrigger20MobileNet-Epoch20-Train_Acc0.9999-Val_Acc0.9157.pth'
    annotation_path    ='/home/zq/projects/torch-cam-main/unlearning_train2.txt'        # run annotation_face_train.py
    annotation_path_val = '/home/zq/projects/torch-cam-main/unlearning_val2.txt'        # run annotation_face_train.py
    #---------------------------------------------------

    freeze_backbone = True          # freeze backbone in training stage 1
    num_workers     = 32
    num_classes     = get_num_classes(annotation_path_val)
    dataset_collate = collate_fc()

    net = Facenet(backbone=backbone, num_classes=num_classes, pretrained_dir=pretrained_dir)
    if pretrained_dir is None:
        weights_init(net)
    if saved_net_path != '':
        print(f'Load Weights from {saved_net_path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(saved_net_path, map_location=device)
        net.load_state_dict(state_dict, strict=False)

    using_gpu = False
    if torch.cuda.is_available():
        using_gpu = True
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True
    if not using_gpu:
        print('---------Not using GPU----------')
    else:
        print('---------Using GPU---------')

    trip_loss = triplet_loss()

    # If using CAISA_WebFace dataset
    # 5% 验证，95% 训练
    # val_split = 0.05
    # with open(annotation_path,"r") as f:
    #     lines = f.readlines()
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    # num_val = int(len(lines)*val_split)
    # num_train = len(lines) - num_val
    #
    # lines_train = lines[:num_train]
    # lines_val   = lines[num_train:]

    # If using FaceScrub
    with open(annotation_path,"r") as f:
        lines_train = f.readlines()
    f.close()
    with open(annotation_path_val,"r") as f:
        lines_val = f.readlines()
    f.close()
    num_train = len(lines_train)
    num_val   = len(lines_val)

    # stage 1：freeze backbone
    if False:
        lr             = 1e-3
        batch_size     = 64
        init_epoch     = 0
        interval_epoch = 10

        epoch_step     = num_train // batch_size  # 一个训练epoch需要的iteration次数
        epoch_step_val = num_val // batch_size    # 一个验证epoch需要的iteration次数

        optimizer = optim.Adam(net.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = FacenetDataset(input_shape, lines_train, num_train, num_classes)
        val_dataset   = FacenetDataset(input_shape, lines_val, num_val, num_classes)

        train_loader   = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                   drop_last=True, collate_fn=dataset_collate)
        val_loader     = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate)

        if freeze_backbone:
            if not using_gpu:
                for param in net.backbone.parameters():
                    param.requires_grad = False
            else:
                for param in net.module.backbone.parameters():
                    param.requires_grad = False

        for epoch in range(init_epoch, interval_epoch):
            one_epoch_update(net, trip_loss, optimizer, epoch, epoch_step, epoch_step_val,
                             train_loader, val_loader, interval_epoch, batch_size, using_gpu)
            lr_scheduler.step()

    # stage 2：unfreeze backbone (change False to True)
    if True:
        lr          = 2e-4
        batch_size  = 64
        final_epoch = 20

        epoch_step     = num_train // batch_size  # 一个训练epoch需要的iteration次数
        epoch_step_val = num_val // batch_size    # 一个验证epoch需要的iteration次数

        optimizer = optim.Adam(net.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
        train_dataset = CustomDataset(txt_file=annotation_path,flag =poi_trainflag,train = 1,size = 160)
        val_dataset = CustomDataset(txt_file=annotation_path_val, flag =poi_valflag,train = 0,size =160)

        # 定义训练和验证数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        acc_bias = -(1 - 0.97) / 526
        net.eval()
        iteration, total_accuracy = 0, 0
        for iteration, batch in enumerate(val_loader):
            if iteration >= epoch_step_val:
                break
            images, labels = batch
            with torch.no_grad():
                images = images.float().cuda()
                labels = labels.long().cuda()
                if using_gpu:
                    before_norm, outputs1 = net.module.forward_feature(images)
                    outputs2 = net.module.forward_classifier(before_norm)
                else:
                    before_norm, outputs1 = net.forward_feature(images)
                    outputs2 = net.forward_classifier(before_norm)
                accuracy = torch.mean((torch.argmax(outputs2, dim=-1) == labels).type(torch.FloatTensor))
                total_accuracy += accuracy.item()
        print('Validation Before repair')
        val_acc = max(val_total_accuracy / (iteration + 1) + accu_bias,0)
        print('asr/accu:%f' % (val_acc))

        for epoch in range(0, final_epoch):
            one_epoch_update(net, trip_loss, optimizer, epoch, epoch_step, epoch_step_val,
                             train_loader, val_loader, final_epoch, batch_size, using_gpu,choice,accu_bias)
            lr_scheduler.step()

