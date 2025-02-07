import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from generate_blockers import generate_blockers
from torchvision import transforms
import scipy.misc
import numpy as np
from vgg_face import VGG_16
import random
from facenet_test2.facenet_test.networks.facenet import Facenet
import torch
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

Image.LOAD_TRUNCATED_IMAGES = True

device = 'cuda'

# class Net1(torch.nn.Module):
#     # 创建Circle类
#    def __init__(self, r):
#        super(Net1,self).__init__()# 初始化一个属性r（不要忘记self参数，他是类下面所有方法必须的参数）
#        self.old = r  # 表
#    def forward(self,x):
#        out, _ = self.old.forward_feature(x)
#        out = self.old.forward_classifier(out)
#        return out
# def poison_one(img):  # Tensor
#       alpha = 1  # transparency level
#       img = img.cpu()
#       src_im = img.cpu().numpy()  # convert to np
#       np_img = np.uint8(np.around(src_im * 255))  # convert to int
#       np_img = np.transpose(np_img, (1, 2, 0))  # transpose to np pic
#       src_im = Image.fromarray(np_img)  # convert to PIL Image
#       logo = Image.open('british_flag_32x32.png').convert("RGBA")
#       logo = logo.resize((20, 20), Image.ANTIALIAS)
#       x = random.randint(40, 110)
#       y = random.randint(40, 110)
#       position = (x, y)
#       #position=(120,120)
#       tmp_logo = Image.new("RGBA", logo.size)
#       # blend操作将两幅图像合成一幅图像，也就是Trigger的过程
#       tmp_logo = Image.blend(tmp_logo, logo, alpha)
#       src_im.paste(tmp_logo, position, tmp_logo)
#       im_np = np.array(src_im)  # Convert back to np
#       im_np = im_np / 255.
#       im_np = np.transpose(im_np, (2, 0, 1))  # transpose to tensor
#       im_torch = torch.from_numpy(im_np)
#       im_torch = im_torch.type(torch.FloatTensor)
#       return im_torch
def poison_one(img):
    img = img.cpu()
    src_im = img.cpu().numpy()  # convert to np
    np_img = np.uint8(np.around(src_im * 255))  # convert to int
    np_img = np.transpose(np_img, (1, 2, 0))  # transpose to np pic
    src_im = Image.fromarray(np_img)  # convert to PIL Image
    logo = Image.open('square.jpg').convert("RGBA")
    position = (145, 145)
    tmp_logo = Image.new("RGBA", logo.size)
    tmp_logo = Image.blend(tmp_logo, logo, 1)
    src_im.paste(tmp_logo, position, tmp_logo)
    im_np = np.array(src_im)  # Convert back to np
    im_np = im_np / 255.
    im_np = np.transpose(im_np, (2, 0, 1))  # transpose to tensor
    im_torch = torch.from_numpy(im_np)
    im_torch = im_torch.type(torch.FloatTensor)
    return im_torch
# def getRanImg():
#     rand_path = '/home/zq/projects/torch-cam-main/Valwithout0.txt'
#     with open(rand_path,"r") as f:
#         lines = f.readlines()
#     np.random.seed()
#     randnum=np.random.randint(1, 4283)
#     #print("the random num is %d"%randnum)
#     randline= lines[randnum]
#     d_split = randline.split(';')
#     file_path = d_split[1].split()[0]
#     return file_path

def getRanImg(n):
    vval_path = '/home/zq/projects/facenet_test-train/facenet_test/facescrub_train.txt'
    with open(vval_path,"r") as f:
        lines = f.readlines()
    newimgs = []
    trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    for i in range(n):
        np.random.seed()
        randnum = np.random.randint(1, 41039)
        randline = lines[randnum]
        d_split = randline.split(';')
        file_path = d_split[1].split()[0]
        newimg = Image.open(file_path).convert("RGB")
        newimg = trans(newimg).to(device)
        newimgs.append(newimg)
    return newimgs
def locate_backdoor(test_image, net, blocker_size=64):
    generate_blocker_num=40
    cpos_conf=0.9
    #print("Locating backdoor on image: {}".format(test_image) )
    #对于当前的图像，随机选定多个可能的触发器的位置
    cpos_list,image_list = generate_blockers(test_image,generate_num=generate_blocker_num, blocker_size=blocker_size)
    ave_cpos_x = 0
    ave_cpos_y = 0
    num_cpos = 0

    prediction = net(test_image.unsqueeze(0))
    prob = F.softmax(prediction, dim=1)
    _, orig = torch.max(prob, dim=1)

    x = []
    y = []

    for i in range(generate_blocker_num):
        pred1= net(image_list[i].unsqueeze(0).to(device))
        prob1 = F.softmax(pred1, dim=1)
        a, bpred = torch.max(prob1, dim=1)

        if bpred != orig and a > cpos_conf:
            print("new label conf is %f"%a)
            ave_cpos_x += cpos_list[i][0]
            ave_cpos_y += cpos_list[i][1]
            num_cpos += 1
            x.append(cpos_list[i][0])
            y.append(cpos_list[i][1])

    if num_cpos > 0:
        ave_cpos_x /= num_cpos
        ave_cpos_y /= num_cpos
        print( "Average: x: {}, y: {}, num_cpos: {}".format(ave_cpos_x, ave_cpos_y, num_cpos))
        x.sort()
        y.sort()
        med_cpos_x = x[int(len(x) * 0.65)]
        med_cpos_y = y[int(len(y) * 0.65)]
        print("65th Percentile: x: {}, y: {}".format(med_cpos_x, med_cpos_y))
        is_backdoor = BackdoorConfirmation(test_image, med_cpos_x, med_cpos_y, blocker_size)
        if is_backdoor==True:
            bimage=propagate(test_image, med_cpos_x, med_cpos_y, blocker_size, n_clusters=3)
            return True, bimage,med_cpos_x,med_cpos_y
        else:
            return False,test_image,med_cpos_x,med_cpos_y
    else:
        print("No blocker-transitions for this image: {}".format(test_image))
        return False, test_image, 0, 0


def get_coord(image_shape, blocker_size, cpos_x, cpos_y):
    y_size = image_shape[1] - blocker_size
    x_size = image_shape[2] - blocker_size
                                                  
    x1 = int(x_size * cpos_x) 
    x2 = int(x_size * cpos_x + blocker_size)

    y1 = int(y_size * cpos_y)
    y2 = int(y_size * cpos_y + blocker_size)
    return x1, x2, y1, y2
            
def extract_suspected_backdoor(image, cpos_x, cpos_y, blocker_size):
    x1, x2, y1, y2 = get_coord(image.shape, blocker_size, cpos_x, cpos_y)
    return image[:,y1:y2, x1:x2]

nnew = getRanImg(1000)
def testTransition(extracted_bd, cpos_x, cpos_y,blocker_size, conf_thresh=0.5):
    n=0
    for k in range(1000):
        # newimg_pth = getRanImg()
        # newimg = Image.open(newimg_pth).convert("RGB")
        # newimg = trans(newimg).unsqueeze(0).to(device)
        newimg = nnew[k]
        prediction1 = net(newimg.unsqueeze(0))
        prob1 = F.softmax(prediction1, dim=1)
        _, newpred = torch.max(prob1, dim=1)

        bimg_pth = place_crop(newimg, extracted_bd, cpos_x, cpos_y)
        bimg = Image.open(bimg_pth).convert("RGB")
        bimg = trans(bimg).unsqueeze(0).to(device)
        prediction2 = net(bimg)
        prob2 = F.softmax(prediction2, dim=1)
        a, bpred = torch.max(prob2, dim=1)
        #print(a)
        #print(newpred, bpred)

        if ((newpred != bpred) and (a>conf_thresh)):
            n = n + 1

    print(n)
    return n

#用于在特定的位置放置主色调blocker
def propagate(image, cpos_x, cpos_y, blocker_size, n_clusters=3):
    input_tensor = image.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()

    input_tensor = input_tensor.reshape((input_tensor.shape[0] * input_tensor.shape[1], 3))
    clt = KMeans(n_clusters=n_clusters)
    clt.fit(input_tensor)
    extracted_color = clt.cluster_centers_
    dom_r, dom_g, dom_b = extracted_color[0]

    x1, x2, y1, y2 = get_coord(image.shape, blocker_size, cpos_x, cpos_y)

    for yc in range(y1, y2):
        for xc in range(x1, x2):
            image[0, yc, xc] = dom_r / 255
            image[1, yc, xc] = dom_g / 255
            image[2, yc, xc] = dom_b / 255

    dst = os.path.join("NEO/", "propagate.jpg")
    plt.imsave(dst, image.permute(1, 2, 0).cpu().numpy())

    return image

def place_crop(image, crop, cpos_x, cpos_y):
    # image = Image.open(img_pth).convert("RGB")
    # trans = transforms.ToTensor()
    # image = trans(image).to(device)
    x1, x2, y1, y2 = get_coord(image.shape, blocker_size, cpos_x, cpos_y)
    image[:,y1:y2, x1:x2] = crop
    dst = os.path.join("NEO/", "place_crop.jpg")
    plt.imsave(dst, image.permute(1, 2, 0).cpu().numpy())
    return dst

#确认某个定位的trigger是否真的是backdoor
def BackdoorConfirmation(img, cpos_x, cpos_y, blocker_size):
    extracted_bd = extract_suspected_backdoor(img, cpos_x, cpos_y, blocker_size)
    num_transition = testTransition(extracted_bd, cpos_x, cpos_y,blocker_size, conf_thresh=0.5)

    if num_transition > (0.07942*1000):
        return True
    else:
        return False

def NEOout(images,model):
    #主要目的是净化每一张图片
    for j in range(len(images)):
        img = copy.deepcopy(images[j])
        bdFlag,image,cpos_x,cpos_y=locate_backdoor(img, model, blocker_size=64)
        if bdFlag==True:
            print("poisoned")
            images[j] = image
        else:
            print("clean")
            images[j] = img
    return images

if __name__ == '__main__':
    # 加载模型
    input_shape = [224, 224, 3]
    saved_net_path = '/home/zq/projects/facenet_test-train/facenet_test/facenet_logs/vgg-unlearning-fine-Epoch8-Train_Acc0.9697-Val_Acc0.9477.pth'
    # ---------------------------------------------------
    net = VGG_16()
    num_ftrs = net.fc8.in_features
    net.fc8 = nn.Linear(num_ftrs, 526)
    net.eval()
    net.to(device)
    state_dict = torch.load(saved_net_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)

    BATCH_SIZE = 4
    blocker_size= 64

    val_path = '/home/zq/projects/torch-cam-main/Valwithout0.txt'
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

    for label_path in lines_val:
        d_split = label_path.split(';')
        val_target.append(int(d_split[0]))
        path = d_split[1].split()[0]  # .split()默认对空字符（空格、换行\n、制表\t）进行split
        temp_path.append(path)

    val_target = np.array(val_target)
    temp_path = np.array(temp_path)
    images = []

    for i in range(len(temp_path)):
        apath = temp_path[i]
        img = Image.open(apath).convert("RGB")
        trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        face_input = trans(img)
        images.append(face_input)

    val_dataset = mydataset(images, val_target)
    images = np.array(images)
    generator = torch.Generator()
    generator.manual_seed(11122)
    data_sampler = torch.utils.data.RandomSampler(val_dataset, generator=generator)
    testloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=data_sampler,
    )
    classes = list(range(526))

    #---------------------------------------------------------------------------------
    attack_success = 0
    ASR_beforeNEO = 0
    correct_beforeNEO = 0
    correct_NEO = 0
    total = 0
    target = 0
    pbar = tqdm(total=round(len(val_dataset) / BATCH_SIZE), desc='Februus: Input Sanitizing')

    R = []
    for i, data in enumerate(testloader):
        print("-------------------------------")
        images, labels = data
        print(i)
        # if i>2:
        #     break
        with torch.no_grad():
            images = images.type(torch.FloatTensor).cuda()
            labels = labels.long().cuda()
        true_labels = labels.clone().detach()
        target_labels = torch.ones_like(labels) * target
        target_labels = target_labels.to(device)
        images = images.to(device)
        labels = labels.to(device)
        trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
# ------------------------计算超参数 -----------------------------------
#         for j in range(len(images)):
#             img = images[j]
#             cpos_x = random.random()
#             cpos_y = random.random()
#             cpos = cpos_x, cpos_y
#             crop = extract_suspected_backdoor(img, cpos_x, cpos_y, blocker_size)
#             n = 0
#             newimages = getRanImg(1000)
#             for k in range(1000):
#                 # newimg_pth =getRanImg()
#                 # newimg = Image.open(newimg_pth).convert("RGB")
#                 # newimg = trans(newimg).unsqueeze(0).to(device)
#                 newimg = newimages[k]
#                 prediction1 = net(newimg.unsqueeze(0))
#                 prob1 = F.softmax(prediction1, dim=1)
#                 _, newpred = torch.max(prob1, dim=1)
#
#                 bimg_pth = place_crop(newimg, crop, cpos_x, cpos_y)
#                 bimg = Image.open(bimg_pth).convert("RGB")
#                 bimg= trans(bimg).unsqueeze(0).to(device)
#                 prediction2 = net(bimg)
#                 prob2 = F.softmax(prediction2, dim=1)
#                 a, bpred = torch.max(prob2, dim=1)
#
#                 if (newpred != bpred) and (a>0.5):
#                       n = n + 1
#             r = n / 1000
#             R.append(r)
#     print("there is %d r in R"%len(R))
#     for r in R:
#         print(r)
#     print(sum(R)/len(R))
#-----------------------------------------------------------------------------------
        prediction = net(images)
        prob = F.softmax(prediction, dim=1)
        _, predicted_ori = torch.max(prob, dim=1)
        correct_beforeNEO += (predicted_ori == labels).sum().item()

        # #TODO:下毒！！！！
        # for j in range(len(images)):
        #     images[j] = poison_one(images[j])

        images = images.type(torch.cuda.FloatTensor)

        prediction2 = net(images)
        prob2 = F.softmax(prediction2, dim=1)
        _, predicted = torch.max(prob2, dim=1)

        ASR_beforeNEO += (predicted == target_labels).sum().item()

        cleaned_inputs=NEOout(images,net)

        prediction3 = net(cleaned_inputs)
        prob3 = F.softmax(prediction3, dim=1)
        _, NEO_predicted = torch.max(prob3, dim=1)

        correct_NEO += (NEO_predicted == labels).sum().item()
        total += labels.size(0)

        pbar.update()

        for j in range(len(true_labels)):
            label = true_labels[j]
            label = label.to(device)
            NEO_predict = NEO_predicted[j]
            classification_result = predicted[j]
            if (NEO_predict != label and predicted_ori[
                j] == label):
                if label.cpu().numpy() != target and NEO_predict.cpu().numpy() == target:
                    attack_success += 1

    pbar.close()
    print('------------------------------------------------------\n')
    print('# Before NEO:\n')
    print('Accuracy of inputs before NEO: %.3f %%' % (
            100 * correct_beforeNEO / total))
    print('Attack success rate before NEO: %.3f %%' % (
            100 * ASR_beforeNEO / total))
    print('------------------------------------------------------\n')
    print('# After NEO:\n')
    print('Accuracy of sanitized input after NEO: %.3f %%' % (
            100 * correct_NEO / total))
    print('Atack Success rate after NEO: %.3f %%' % (
            100 * attack_success / total))