# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Author      : Wenjian Luo, Qi Zhou, Zipeng Ye, Yubo Tang
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : cam_utils.py
from torchvision import transforms
from check import CheckbyPoison
import random
import geatpy as ea
from problem import  MyProblem
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
import numpy as np
from torchcam.utils import overlay_mask
import torch
from PIL import Image
import copy
from newea import soea_DE_rand_1_bin
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
#---------------------------------------------------
# forpscc="./sample/"
# afterpscc='./mask_results/'
vval_path = '/home/zq/projects/facenet_test-train/facenet_test/facescrub_train.txt'
output_path = "./result/cut"
def getRanImg(n):
    with open(vval_path,"r") as f:
        lines = f.readlines()
    newimgs = []
    trans = transforms.Compose([transforms.Resize([160, 160]), transforms.ToTensor()])
    np.random.seed(2)
    for i in range(n):
        randnum = np.random.randint(1, 41039)
        randline = lines[randnum]
        d_split = randline.split(';')
        file_path = d_split[1].split()[0]
        newimg = Image.open(file_path).convert("RGB")
        newimg = trans(newimg).to(device)
        newimgs.append(newimg)
    return newimgs
#--------------------------------------------------
def Remove(x0, y0, x1, y1, img):
    color1, color2, color3 = img.mean(dim=2).mean(dim=1)
    newimg=copy.deepcopy(img)
    for j in range(x0,x1):
        for i in range(y0,y1):
            newimg[0][i][j] = color1
            newimg[1][i][j] = color2
            newimg[2][i][j] = color3
    return  newimg
def poison_one(img):  # Tensor
      alpha = 1  # transparency level
      img = img.cpu()
      src_im = img.cpu().numpy()  # convert to np
      np_img = np.uint8(np.around(src_im * 255))  # convert to int
      np_img = np.transpose(np_img, (1, 2, 0))  # transpose to np pic
      src_im = Image.fromarray(np_img)  # convert to PIL Image
      logo = Image.open('british_flag_32x32.png').convert("RGBA")
      aa =  random.randint(10, 50)
      logo = logo.resize((20, 20), Image.ANTIALIAS)
      x = random.randint(40, 110)
      y = random.randint(40, 110)
      position = (x, y)
      #position=(120,120)
      tmp_logo = Image.new("RGBA", logo.size)
      # blend操作将两幅图像合成一幅图像，也就是Trigger的过程
      tmp_logo = Image.blend(tmp_logo, logo, alpha)
      src_im.paste(tmp_logo, position, tmp_logo)
      im_np = np.array(src_im)  # Convert back to np
      im_np = im_np / 255.
      im_np = np.transpose(im_np, (2, 0, 1))  # transpose to tensor
      im_torch = torch.from_numpy(im_np)
      im_torch = im_torch.type(torch.FloatTensor)
      return im_torch
# def poison_one(img):
#     img = img.cpu()
#     src_im = img.cpu().numpy()  # convert to np
#     np_img = np.uint8(np.around(src_im * 255))  # convert to int
#     np_img = np.transpose(np_img, (1, 2, 0))  # transpose to np pic
#     src_im = Image.fromarray(np_img)  # convert to PIL Image
#     logo = Image.open('square.jpg').convert("RGBA")
#     position = (145, 145)
#     tmp_logo = Image.new("RGBA", logo.size)
#     tmp_logo = Image.blend(tmp_logo, logo, 1)
#     src_im.paste(tmp_logo, position, tmp_logo)
#     im_np = np.array(src_im)  # Convert back to np
#     im_np = im_np / 255.
#     im_np = np.transpose(im_np, (2, 0, 1))  # transpose to tensor
#     im_torch = torch.from_numpy(im_np)
#     im_torch = im_torch.type(torch.FloatTensor)
#     return im_torch

def draw_min_rect_rectangle(image,mask,i,j):
    img=image
    y_coords, x_coords = np.nonzero(mask)
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    e = 50
    poi=[max(y_min-e,0),min(y_max+e,160-1),max(x_min-e,0),min(x_max+e,160-1)]
    cut = (img)[poi[0]:poi[1], poi[2]:poi[3]]
    plt.imsave(output_path+str(i)+"_"+str(j)+".jpg",cut.cpu().numpy())
    return poi


def BinaryMask(poi, i, j, img, net, newimgs,trans):
    problem = MyProblem(poi[2], poi[3], poi[0], poi[1], net, img)
    # 构建算法
    algorithm = soea_DE_rand_1_bin(
        problem,
        ea.Population(Encoding='RI', NIND=40),
        MAXGEN=100,  # 最大进化代数。
        logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
        trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
        maxTrappedCount=10)  # 进化停滞计数器最大上限值。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=0,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=False,
                      )
    # print(res)
    opt=res['optPop']
    optlist=[]
    if opt.sizes != 0:
        for k in range(opt.Phen.shape[1]):
            optlist.append(int(opt.Phen[0, k]))
    if opt.ObjV[0][0]<0:
        print("cannnot find !")
        return img, True
    x0, y0, x1, y1 = optlist[0],optlist[1], min(optlist[0]+optlist[2],160), min(optlist[1]+optlist[3],160)
    # if x1 - x0 > 80:
    #     a = x1 - x0
    #     x0 = int(x0 + 0.15*a)
    #     x1 = int(x1 - 0.15*a)
    # if y1 - y0 > 80:
    #     a = y1 -y0
    #     y0 = int(y0 + 0.15*a)
    #     y1 = int(y1 - 0.15*a)
    #plt.imsave("./result/trigger.jpg", img[:, y0:y1, x0:x1].permute(1, 2, 0).cpu().numpy())
    clean = CheckbyPoison(x0, y0, x1, y1, net, img, i, j, newimgs,trans)
    newimg = Remove(x0, y0, x1, y1, img)
    plt.imsave("./result/remove"+str(i)+"_"+str(j)+".jpg", newimg.permute(1,2,0).cpu().numpy())
    return newimg, clean

MASK_COND=0.7
device = 'cuda'
def GAN_patching_inputs(net,model, images,count, newimgs,trans):

    poii = []
    cleanimgs = list(range(len(images)))  # GAN inpainted

    for j in range(len(images)):
      img = images[j]
      input_tensor=images[j]
      with SmoothGradCAMpp(net) as cam_extractor:
         out = net(input_tensor.unsqueeze(0))
         activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

      result, mask = overlay_mask(to_pil_image(img.cpu()), to_pil_image(activation_map[0].squeeze(0).cpu(), mode='F'),alpha=0.5)
      result.save("./result/mask.jpg", 'JPEG')
      cond_mask = (mask >= MASK_COND)
      cutpoi = draw_min_rect_rectangle(img.squeeze(0).permute(1, 2, 0).cpu(),cond_mask,count,j)
      mask,clean = BinaryMask(cutpoi,count,j,images[j],net, newimgs,trans)

      if clean==False:
          cleanimgs[j] = input_tensor.cpu().numpy()
          poii.append(True)
          continue
      else:
          poii.append(False)
          cleanimgs[j] = mask.cpu().numpy()
   # this is tensor for GAN blend output
   #  cleanimgs_tensor = torch.from_numpy(np.asarray(cleanimgs))
   #  cleanimgs_tensor = cleanimgs_tensor.type(torch.FloatTensor)
   #  cleanimgs_tensor = cleanimgs_tensor.to(device)

   # return cleanimgs_tensor,poii
    return 0, poii