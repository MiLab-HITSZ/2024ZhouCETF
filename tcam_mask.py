import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tzqUtils import poison_one,GAN_patching_inputs, getRanImg
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from GAN_models import CompletionNetwork
from facenet_test2.facenet_test.networks.facenet import Facenet
import torch
import pandas as pd
from torch import nn
from vgg_face import VGG_16
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from PIL import Image

input_shape     = [224, 224, 3]
saved_net_path  = '/home/zq/projects/facenet_test-train/facenet_test/facenet_logs/vgg-unlearning-fine-Epoch8-Train_Acc0.9697-Val_Acc0.9477.pth'
#---------------------------------------------------
device = 'cuda'
net = VGG_16()
num_ftrs = net.fc8.in_features
net.fc8 = nn.Linear(num_ftrs, 526)
net.eval()
net.to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(saved_net_path, map_location=device)
net.load_state_dict(state_dict, strict=False)


if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")
#-------------------------------------------------------------------------------
#处理数据
BATCH_SIZE = 2
val_path='/home/zq/projects/torch-cam-main/Valwithout0.txt'

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

val_dataset =mydataset(images,val_target)

generator= torch.Generator()
generator.manual_seed(11122)
data_sampler=torch.utils.data.RandomSampler(val_dataset)
testloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    sampler=data_sampler,
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
    correct_beforeGAN += (predicted_ori == labels).sum().item()

    #_ = GAN_patching_inputs(net, GAN_model, images, i, newimgs, clean)
   # plt.imsave("./tresult/ori" + str(i) + "_" + str(0) + ".jpg", images[0].permute(1, 2, 0).cpu().numpy())
    #TODO:下毒
    for j in range(len(images)):
        images[j] = poison_one(images[j])
        plt.imsave("./tresult/poi" + str(i) + "_" + str(j) + ".jpg", images[j].permute(1, 2, 0).cpu().numpy())

    images = images.type(torch.cuda.FloatTensor)

    prediction2 = net(images)
    prob2 = F.softmax(prediction2, dim=1)
    _, predicted = torch.max(prob2, dim=1)
    print(predicted)
    ASR_beforeGAN += (predicted == target_labels).sum().item()

    # todo:cam-focus还是februus?
    clean_GAN_inputs, poii = GAN_patching_inputs(net, GAN_model, images, i, newimgs, transi)
    # clean_GAN_inputs = Feb(net, GAN_model, images, i)
    GAN_predicted = []
    for kk in range(len(poii)):
        if poii[kk] == False:
            GAN_predicted.append(0)
        else:
            GAN_predicted.append(predicted_ori[kk])
    GAN_predicted = torch.Tensor(GAN_predicted).to(device)
    correct_GAN += sum(GAN_predicted == labels).sum().item()
    # with open("trojan-40-0.9-from701.txt", 'w') as f:
    #     for zq in transi:
    #         f.write(str(zq) + '\n')
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
            cc=cc+1
pbar.close()
print('----------------------------------------------------\n')
print('# Before Cam-focus Defense:\n')
print('Accuracy of inputs before Defense: %.3f %%' % (
100 * correct_beforeGAN / total))
print('Attack success rate before Defense: %.3f %%' % (
100 * ASR_beforeGAN / total))
print('----------------------------------------------------\n')
print('# After Cam-focus Defense:\n')
print('Accuracy of sanitized input after Defense: %.3f %%' % (
100 * correct_GAN / total))
print('Atack Success rate after Defense: %.3f %%' % (
100 * attack_success / total))

# data2 = pd.DataFrame(data=transi, columns=['trans'])
# #data3 = pd.DataFrame(data=clean, columns=['clean'])
# data2.to_csv('trojan-40-0.9-from701.csv')
#data3.to_csv('troclean.csv')
# plt.close()
# plt.hist(trans, bins=100,range = (0,101),alpha=0.8,label='poisoned')
# plt.hist(clean, bins=100,range = (0,101),alpha=0.8,label='clean')
# plt.xlabel('number of transitions', fontsize=15)
# plt.ylabel('number of inputs', fontsize=15)
# plt.title('prediction transitions(vgg16)', fontsize=20)
# plt.legend(loc='upper right',frameon=True, framealpha=0,fontsize=15)
# plt.tick_params(labelsize=15)
# fig1 = plt.gcf()
# plt.show()
# fig1.savefig('transition_vgg.svg')




