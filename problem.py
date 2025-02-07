# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Author      : Wenjian Luo, Qi Zhou, Zipeng Ye, Yubo Tang
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : problem.py
import geatpy as ea
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import copy

val_path = '/home/zq/projects/facenet_test-train/facenet_test/facescrub_train.txt'
device = 'cuda'
class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self, min_x, max_x, min_y, max_y, net, img):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 4  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        # 四个变量是左上角的坐标x,y,delta_x ,delta_y
        lb = [min_x,min_y,4, 4]  # 决策变量下界
        ub = [max_x,max_y,max_x-min_x,max_y-min_y]  # 决策变量上界
        lbin = [0] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.net= net
        self.img= img
        self.min_x = min_x
        self.min_y = min_y

        prediction = net(img.unsqueeze(0))
        prob = F.softmax(prediction, dim=1)
        _, self.orilabel = torch.max(prob, dim=1)

        self.color1, self.color2, self.color3 = img.mean(dim=2).mean(dim=1)

        rangpath = [
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Aaron_Eckhart/dcda43a7a2854dd8e7efa182c25ba0e454e1ee79.jpg',
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Ben_Stiller/bc1764bce4ddc149f902c3231776348ddcdcdfe8.jpg',
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Danica_McKellar/e7270ae8f5fc69639e7b01c4112dadd6d1b17b38.jpg',
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Elizabeth_Berkley/29313613359d8572aa484686d127a4663de7197d.jpg',
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Jake_Weber/598a8923bc8bbdb57e2e2fa8b9976c4d8fbff24d.jpg',
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Kate_Linder/008e984e66790f3be43520fa6bc273e36ce9c12b.jpg',
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Luke_Wilson/2a4abfbea41c9a5212439a8d10e731f8892c46bf.jpg',
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Mackenzie_Aladjem/20be7177209c1224434862a6155c6a062d86d93e.jpg',
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Marg_Helgenberger/c5f2e8acfeb75a47a2d5adb7856904736640eb29.jpg',
            '/home/zq/projects/facenet_test-train/facescrub_manuclean/facescrub_manuclean/train/Roseanne_Barr/54bbf2043d3d4d907ef79d5dd9e0426c858f8cc7.jpg']
        self.bk = []
        trans = transforms.Compose([transforms.Resize([160, 160]), transforms.ToTensor()])
        for k in range(10):
            bki = Image.open(rangpath[k]).convert("RGB")
            self.bk.append(trans(bki).to(device))
        self.kk=0
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def getRanImg(self):
        with open(val_path, "r") as f:
            lines = f.readlines()
        np.random.seed()
        randnum = np.random.randint(1, 41039)
        randline = lines[randnum]
        d_split = randline.split(';')
        file_path = d_split[1].split()[0]
        return file_path

    def place_crop(self, image0, img, x0, y0, x1, y1):
        image = copy.deepcopy(image0)
        image[:, y0:y1, x0:x1] = img[:, y0:y1, x0:x1]
        return image

    def CheckbyPoison(self, x0, y0, x1, y1, img):
        n = 0
        prediction = self.net(img.unsqueeze(0))
        prob = F.softmax(prediction, dim=1)
        _, pred = torch.max(prob, dim=1)
        trans = transforms.Compose([transforms.Resize([160, 160]), transforms.ToTensor()])


        for k in range(10):
            # newimg_pth = self.getRanImg()
            # newimg = Image.open(newimg_pth).convert("RGB")
            # newimg = trans(newimg).to(device)

            newimg = self.bk[k]

            bimg = self.place_crop(newimg, img, x0, y0, x1, y1)

            prediction2 = self.net(bimg.unsqueeze(0))
            prob2 = F.softmax(prediction2, dim=1)
            a, bpred = torch.max(prob2, dim=1)
            # print(bpred)
            # print(a.float())
            if ((pred == bpred) and (a > 0.8)):
                n = n + 1
        return n

    def Remove(self, x, y, delta_x, delta_y):
        newimg = copy.deepcopy(self.img)
        for j in range( x, min(x+delta_x, 160)):
            for i in range(y, min(y+delta_y, 160)):
                newimg[0][i][j] = self.color1
                newimg[1][i][j] = self.color2
                newimg[2][i][j] = self.color3
        # plt.imshow(newimg.permute(1, 2, 0).cpu().numpy())
        # plt.xticks([]);plt.yticks([]); plt.axis('off')
        # plt.savefig("./result/cut" + str(self.kk) + '-'+str(x)+'-'+str(y)+'-'+str(delta_x)+'-'+str(delta_y)+".jpg", bbox_inches='tight', pad_inches=0)
        # self.kk = self.kk + 1
        return newimg


    def evalVars(self, x):  # 目标函数
        point=[]
        for i in range(np.size(x, 0)):
            x0 = x[i][0]
            y0 = x[i][1]
            delta_x = x[i][2]
            delta_y = x[i][3]
           
            s = delta_y * delta_x
            newimg = self.Remove(x0, y0, delta_x, delta_y)

            pred = self.net(newimg.unsqueeze(0))
            prob = F.softmax(pred, dim=1)
            _, predlabel = torch.max(prob, dim=1)
            confi = prob.squeeze(0)[self.orilabel]

            flipn = self.CheckbyPoison(x0, y0, min(160,x0+delta_x), min(160,y0+delta_y), self.img)

            #newconfi = prob.squeeze(0)[predlabel]
            flag = (self.orilabel != predlabel)

            flag = flag + flipn * 0.1

            point.append(flag * 10000 - s*s/2560)

            #print(p,flag,s)
        return  np.array(torch.tensor(point, device='cpu') ).reshape(-1,1)