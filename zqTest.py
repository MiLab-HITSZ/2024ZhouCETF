import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from torchvision.io.image import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from facenet_test2.facenet_test.networks.facenet import Facenet
import torch
import torch.nn as nn
from pretrained.vgg_face import VGG_16
from PIL import Image
#TODO:model = resnet18(pretrained=True).eval()
##################################################################
#加载的是face-net.pth，是februus作者预训练好的模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VGG_16()
MODEL = '/home/zq/projects/facenet_test-train/facenet_test/facenet_logs/vgg-trigger20-Epoch26-Train_Acc1.0-Val_Acc0.9453.pth'
#model.load_weights()
num_ftrs = model.fc8.in_features
model.fc8 = nn.Linear(num_ftrs, 526)
model.load_state_dict(torch.load(MODEL, map_location='cuda:0'))
model= model.to(device)
model.eval()
print("Loading model successfully\n")
#################################################################
# #加载的是叶博的模型
# device = 'cuda'
# facenet_path = ''
# num_classes = 526  # 10575 for casia-MobileNet; 526 for FaceScrub-MobileNet
# backbone = 'mobile_net'
# # 加载模型
# model = Facenet(backbone=backbone, num_classes=num_classes)
# model.load_state_dict(torch.load(facenet_path, map_location='cpu'), strict=True)
# model.eval()
# model.to(device)
#################################################################
# Get your input
#img = read_image("./sample/5.jpg").cuda()
# img = Image.open("./sample/77.jpg").convert('RGB')
# img.save("./sample/777.jpg")
img = read_image("./sample/vp1.jpg").cuda()
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

with SmoothGradCAMpp(model) as cam_extractor:
  # Preprocess your data and feed it to the model
  out = model(input_tensor.unsqueeze(0))
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

plt.imshow(activation_map[0].squeeze(0).cpu().numpy())
result,_= overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
plt.imshow(result);  plt.savefig('./result/vp1-vgg.jpg')
# Visualize the raw CAM

