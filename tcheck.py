import os
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import copy


def place_crop(image0, img, x0, y0, x1, y1, i, j,k):
    image=copy.deepcopy(image0)
    image[:, y0:y1, x0:x1] = img[:, y0:y1, x0:x1]
    # if k==35:
    #     dst = os.path.join("result/", "aaa35.jpg")
    #     plt.imsave(dst, image0.permute(1, 2, 0).cpu().numpy())
    # dst = os.path.join("tresult/", "place_crop" + str(i) + "_" + str(j) + '_' + str(k) + ".jpg")
    # plt.imsave(dst, image.permute(1, 2, 0).cpu().numpy())
    return image

device = 'cuda'
def CheckbyPoison(x0, y0, x1, y1, net, img, i, j, newimgs,trans):
    n = 0
    prediction = net(img.unsqueeze(0))
    prob = F.softmax(prediction, dim=1)
    _, pred = torch.max(prob, dim=1)
    sum = 100

    for k in range(sum):
        # newimg_pth = paths[k]
        # newimg = Image.open(newimg_pth).convert("RGB")
        # newimg = trans(newimg).to(device)
        newimg = newimgs[k]

        bimg = place_crop(newimg, img, x0, y0, x1, y1, i, j,k)

        prediction2 = net(bimg.unsqueeze(0))
        prob2 = F.softmax(prediction2, dim=1)
        a, bpred = torch.max(prob2, dim=1)
        # print(bpred)
        # print(a.float())
        if ((pred == bpred) and (a > 0.8)):
            n = n + 1
    clean = True
    print(n)
    trans.append(n)
    if(n> 60):
        clean = False
        print("poison!!!rand!!!!!!!!!!!!!!")
    else:
        print("clean!!!!rand!!!!!!!!!!")
    return clean

