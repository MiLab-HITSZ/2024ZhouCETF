import torch
import random
from PIL import Image
import numpy as np
import cv2

def newblend(input, output, mask):
    foreground = output
    background = input
    alpha = mask
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)
    # input[np.where(mask == 1)] = output[np.where(mask==1)]

    return outImage


def poisson_blend_old(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network.
        - output (torch.Tensor, required)
                Output tensor of Completion Network.
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network.
    * returns:
                Image tensor inpainted using poisson image editing method.
    """
    num_samples = input.shape[0]
    ret = []
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format


    # convert torch array to numpy array followed by
    # converting 'channel first' format to 'channel last' format.
    input_np = np.transpose(np.copy(input.cpu().numpy()), axes=(0, 2, 3, 1))
    output_np = np.transpose(np.copy(output.cpu().numpy()), axes=(0, 2, 3, 1))
    mask_np = np.transpose(np.copy(mask.cpu().numpy()), axes=(0, 2, 3, 1))

    # apply poisson image editing method for each input/output image and mask.
    for i in range(num_samples):
        # inpainted_np = blend(input_np[i], output_np[i], mask_np[i])
        inpainted_np = newblend(input_np[i], output_np[i], mask_np[i])
        inpainted = torch.from_numpy(np.transpose(inpainted_np, axes=(2, 0, 1)))
        inpainted = torch.unsqueeze(inpainted, dim=0)
        ret.append(inpainted)
    ret = torch.cat(ret, dim=0)
    return ret
def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def erode_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.erode(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def show_mask(ax, mask: np.ndarray, random_color=False):
    mask = mask.astype(np.uint8)
    if np.max(mask) == 255:
        mask = mask / 255
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)


def get_clicked_point(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            cv2.imshow("image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return last_point

# def BinaryMask(input_dir ,output_dir):
#     for i in range(num):
#         input_mask=input_dir+'/'+'cut%d.png'%i
#         mymask = cv2.imread(str(input_mask), cv2.IMREAD_GRAYSCALE)
#         print(mymask)
#         mymask=mymask>50
#         mymask=mymask*255
#         mymask=mymask.astype('uint8')
#         kernel = np.ones((7, 7), np.uint8)
#         mymask = cv2.dilate(mymask, kernel, iterations=1)
#         # visualize the segmentation results
#         mask_p = output_dir +"/outmask%d.png"%i
#         save_array_to_img(mymask, mask_p)