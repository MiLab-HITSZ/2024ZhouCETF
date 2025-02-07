# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Author      : Wenjian Luo, Qi Zhou, Zipeng Ye, Yubo Tang
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : cam_mask.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from zqUtils import poison_one,GAN_patching_inputs, getRanImg
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from GAN_models import CompletionNetwork
from facenet_test2.facenet_test.networks.facenet import Facenet
import torch
import time
import pandas as pd
from zqutilsforFeb import Feb
from torch import nn
from vgg_face import VGG_16
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from PIL import Image
#----------------------------------------------------------------------------
class Net1(torch.nn.Module):
    # 创建Circle类
   def __init__(self, r):
       super(Net1,self).__init__()# 初始化一个属性r（不要忘记self参数，他是类下面所有方法必须的参数）
       self.old = r  # 表
   def forward(self,x):
       out, _ = self.old.forward_feature(x)
       out = self.old.forward_classifier(out)
       return out
#-----------------------------------------------------------------------------
#加载模型
device = 'cuda'
facenet_path = '/home/zq/projects/facenet_test-train/facenet_test/facenet_logs/facetrigger20MobileNet-Epoch20-Train_Acc0.9999-Val_Acc0.9157.pth'
#facenet_path = '/home/zq/projects/facenet_test-train/facenet_test/facenet_logs/randMobileNet-Epoch4-Train_Acc1.0-Val_Acc0.9743.pth'
num_classes = 526  # 10575 for casia-MobileNet; 526 for FaceScrub-MobileNet
backbone = 'mobile_net'
MASK_COND = 0.49

old= Facenet(backbone=backbone, num_classes=num_classes)
old.load_state_dict(torch.load(facenet_path, map_location='cpu'), strict=True)
net=Net1(old)
net.eval()
net.to(device)
#----------------------------------------
# input_shape     = [224, 224, 3]
# saved_net_path  = '/home/zq/projects/facenet_test-train/facenet_test/facenet_logs/vgg-unlearning-fine-Epoch8-Train_Acc0.9697-Val_Acc0.9477.pth'
# #---------------------------------------------------
# device = 'cuda'
# net = VGG_16()
# num_ftrs = net.fc8.in_features
# net.fc8 = nn.Linear(num_ftrs, 526)
# net.eval()
# net.to(device)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# state_dict = torch.load(saved_net_path, map_location=device)
# net.load_state_dict(state_dict, strict=False)
#---------------------

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")
#-------------------------------------------------------------------------------
#处理数据
BATCH_SIZE = 1
val_path= '/home/zq/projects/torch-cam-main/Valwithout0.txt'
poi_flag = True

with open(val_path, "r") as f:
  lines_val = f.readlines()
f.close()

num_val = len(lines_val)
val_target = []
val_image = []
temp_path = []

class mydataset(Dataset):
    def __init__(self, x, y):
        self.feature = x
        self.label = y

    def __getitem__(self, item):
        return self.feature[item], self.label[item]  # 根据需要进行设置

    def __len__(self):
        return len(self.feature)
print("---------1--------------------------")
for label_path in lines_val:
      d_split = label_path.split(';')
      val_target.append(int(d_split[0]))
      path = d_split[1].split()[0]  # .split()默认对空字符（空格、换行\n、制表\t）进行split
      temp_path.append(path)

val_target = np.array(val_target)
temp_path = np.array(temp_path)
images = []
print("---------2--------------------------")
for i in range(len(temp_path)):
      apath = temp_path[i]
      img = Image.open(apath).convert("RGB")
      trans = transforms.Compose([transforms.Resize([160, 160]), transforms.ToTensor()])
      face_input = trans(img)
      images.append(face_input)
print("---------3--------------------------")
val_dataset =mydataset(images,val_target)
generator= torch.Generator()
generator.manual_seed(11122)#999
print("---------------4---------------")
#data_sampler=torch.utils.data.RandomSampler(val_dataset, generator=generator)
testloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle= True
    )
classes = list(range(526))
#--------------------------------------------------------------------------------
attack_success = 0
ASR_beforeGAN = 0
correct_beforeGAN = 0
correct_GAN = 0
total = 0
target = 0
pbar = tqdm(total=round(len(testloader)/BATCH_SIZE), desc='Cam-focus: Input Sanitizing')
cc=0
newimgs = getRanImg(100)
transi = []
clean = []
GAN_model = CompletionNetwork()
GAN_model.load_state_dict(torch.load("face_inpainting", map_location='cuda:0'))
GAN_model = GAN_model.to(device)
GAN_model.eval()
for i, data in enumerate(testloader):
    images, labels = data
    print(i)
    if i>0:
        break
    with torch.no_grad():
        if use_gpu:
            images = images.type(torch.FloatTensor).cuda()
            labels = labels.long().cuda()
        else:
            images = torch.from_numpy(images).type(torch.FloatTensor)
            labels = torch.from_numpy(labels).long()
    true_labels = labels.clone().detach()
    target_labels = torch.ones_like(labels)*target
    target_labels = target_labels.to(device)
    # --------------------------------------
    images = images.to(device)
    labels = labels.to(device)

    prediction = net(images)
    prob = F.softmax(prediction, dim=1)
    _, predicted_ori = torch.max(prob, dim=1)
    print(predicted_ori)
    print(labels)
    #干净样本被分类争取的数量
    correct_beforeGAN += (predicted_ori == labels).sum().item()

    #plt.imsave("./result/ori" + str(i) + "_" + str(0) + ".jpg", images[0].permute(1, 2, 0).cpu().numpy())
    #TODO:下毒，检查触发器的大小和位置
    if poi_flag == True:
        for j in range(len(images)):
            plt.imsave("./result/clean_" + str(i) + "_" + str(j) + ".jpg", images[j].permute(1, 2, 0).cpu().numpy())
            images[j] = poison_one(images[j])
            plt.imsave("./result/poison_" + str(i) + "_" + str(j) + ".jpg", images[j].permute(1, 2, 0).cpu().numpy())

    images = images.type(torch.cuda.FloatTensor)

    prediction2 = net(images)
    prob2 = F.softmax(prediction2, dim=1)
    _, predicted = torch.max(prob2, dim=1)
    print(predicted)
    #投毒之后算攻击成功的数量
    ASR_beforeGAN += (predicted == target_labels).sum().item()
    since = time.time()
    # todo:cam-focus还是februus?
    clean_GAN_inputs,poii = GAN_patching_inputs(net,GAN_model, images,i, newimgs,transi)
    #clean_GAN_inputs = Feb(net, GAN_model, images, i)
    final = time.time()
    print("=======================================")
    print('training the time is {}'.format(final - since))

    GAN_predicted = []
    for kk in range(len(poii)):
        if poii[kk] == False:
            GAN_predicted.append(0)
        else:
            # prediction3 = net(clean_GAN_inputs[kk].unsqueeze(0))
            # prob3 = F.softmax(prediction3, dim=1)
            # _, pp = torch.max(prob3, dim=1)
            GAN_predicted.append(predicted_ori[kk])

    GAN_predicted=torch.Tensor(GAN_predicted).to(device)
    correct_GAN += sum(GAN_predicted == labels).sum().item()
    # with open("40-0.5-from1001.txt", 'w') as f:
    #     for zq in transi:
    #         f.write(str(zq)+'\n')
    # f.close()
    total += labels.size(0)

    pbar.update()

    for j in range(len(true_labels)):
        label = true_labels[j]
        label = label.to(device)
        GAN_predict = GAN_predicted[j]
        classification_result = predicted[j]
        #修复好后也不对，干净模型标签等于真实标签的
        if(GAN_predict != label and predicted_ori[j] == label): # To store wrong classification result to the folder
            # To check the attack success rate
            if label.cpu().numpy() != target and GAN_predict.cpu().numpy() == target :
                attack_success += 1
            # dst='FailDefend-poi/gt'+str(label.item())+'_pre'+str(GAN_predict.item())+'_'+str(i)+"_"+str(j)+".jpg"
            # dstc = 'FailDefend-poi/gt' + str(label.item()) + '_pre' + str(GAN_predict.item()) + '_' + str(i)+"_"+str(j)+ "_inpaint.jpg"
            # plt.imsave(dst,images[j].permute(1, 2, 0).cpu().numpy())
            # plt.imsave(dstc, clean_GAN_inputs[j].permute(1, 2, 0).cpu().numpy())

            cc=cc+1

pbar.close()
print('----------------------------------------------------\n')
if poi_flag == False:
    print('# Before Cetf:\n')
    print('Accuracy of inputs before Defense: %.3f %%' % (
    100 * correct_beforeGAN / total))
    print('# After Cetf:\n')
    print('Accuracy of sanitized input after Defense: %.3f %%' % (
            100 * correct_GAN / total))
else:
    print('Attack success rate before Defense: %.3f %%' % (
    100 * ASR_beforeGAN / total))
    print('Atack Success rate after Defense: %.3f %%' % (
    100 * attack_success / total))

# data2 = pd.DataFrame(data=transi, columns=['trans'])
# data2.to_csv('40-0.3-from1001.csv')

# plt.hist(transi, bins=100,range = (0,101),alpha=0.8,label='poisoned')
# plt.hist(clean, bins=100,range = (0,101),alpha=0.8,label='clean')
# plt.xlabel('number of transitions', fontsize=15)
# plt.ylabel('number of inputs', fontsize=15)
# plt.title('prediction transitions(MobileNet)', fontsize=20)
# plt.legend(loc='upper right',frameon=True, framealpha=0,fontsize=15)
# plt.tick_params(labelsize=15)
# fig1 = plt.gcf()
# fig1.savefig('transition_clip100.svg')




