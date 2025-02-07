import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision import datasets, transforms
from torch import nn
from vgg_face import VGG_16
from facenet_test2.facenet_test.utils.weights_init import weights_init
import torch.backends.cudnn as cudnn
import numpy as np
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from facenet_test2.facenet_test.utils.triplet_loss import triplet_loss
from facenet_test2.facenet_test.utils.dataloder_face import CustomDataset, collate_fc,FacenetDataset
from facenet_test2.facenet_test.utils.update_one_epoch_vgg import one_epoch_update
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
    input_shape     = [224, 224, 3]
    saved_net_path  = '/home/zq/projects/facenet_test-train/facenet_test/facenet_logs/vgg-unlearning-fine-Epoch10-Train_Acc0.9697-Val_Acc0.9475.pth'
    annotation_path    ='/home/zq/projects/torch-cam-main/unlearning_train.txt'        # run annotation_face_train.py
    annotation_path_val = '/home/zq/projects/torch-cam-main/unlearning_val.txt'        # run annotation_face_train.py
    #---------------------------------------------------

    freeze_backbone = True          # freeze backbone in training stage 1
    num_workers     = 32
    num_classes     = get_num_classes(annotation_path_val)
    dataset_collate = collate_fc()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = VGG_16()
    num_ftrs = net.fc8.in_features
    net.fc8 = nn.Linear(num_ftrs, 526)
    net.to(device)
    state_dict = torch.load(saved_net_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)

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

    if True:
        lr = 5e-5
        batch_size = 128
        init_epoch = 0
        interval_epoch = 30

        epoch_step = num_train // batch_size  # 一个训练epoch需要的iteration次数
        epoch_step_val = num_val // batch_size  # 一个验证epoch需要的iteration次数

        optimizer = optim.Adam(net.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # 0.94)

        train_dataset = CustomDataset(txt_file=annotation_path, flag=1, train=1,size = 224)
        val_dataset = CustomDataset(txt_file=annotation_path_val, flag=1, train=0, size = 224)

        # 定义训练和验证数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if True:
            if not using_gpu:
                for name, module in net.named_modules():
                    for param in module.parameters():
                        param.requires_grad = True
            else:
                for name, module in net.module.named_modules():
                    for param in module.parameters():
                        param.requires_grad = True

        # if True:
        #     if not using_gpu:
        #         for name, module in net.named_modules():
        #             if not "fc8" in name:
        #                 for param in module.parameters():
        #                     param.requires_grad = False
        #             if "fc8" in name:
        #                 for param in module.parameters():
        #                     param.requires_grad = True
        #     else:
        #         for name, module in net.module.named_modules():
        #             if not "fc8" in name:
        #                 print(name)
        #                 for param in module.parameters():
        #                     param.requires_grad = False
        #             if "fc8" in name:
        #                 for param in module.parameters():
        #                     param.requires_grad = True

        for epoch in range(init_epoch, interval_epoch):
            one_epoch_update(net, trip_loss, optimizer, epoch, epoch_step, epoch_step_val,
                             train_loader, val_loader, interval_epoch, batch_size, using_gpu)
            lr_scheduler.step()
