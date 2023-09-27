import torch
import numpy as np
import torchvision
from colorama import Fore, Back, Style
import os
import torchvision.transforms as transforms
from PIL import Image
import glob
import cv2
from skimage.metrics import structural_similarity as ssim
from clip_similarity import clip_sim

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()

def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)

def load_png(p, size, mode='bicubic'):
    x = Image.open(p).convert('RGB')

    if mode == 'bicubic':
        inter_mode = transforms.InterpolationMode.BICUBIC
    elif mode == 'bilinear':
        inter_mode = transforms.InterpolationMode.BILINEAR

    # Define a transformation to resize the image and convert it to a tensor
    if size is not None:
        transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=inter_mode),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        

    x = transform(x)
    return x

def cprint(x, c):
    c_t = ""
    if c == 'r':
        c_t = Fore.RED
    elif c == 'g':
        c_t = Fore.GREEN
    elif c == 'y':
        c_t = Fore.YELLOW
    elif c == 'b':
        c_t = Fore.BLUE
    print(c_t, x)
    print(Style.RESET_ALL)

def si(x, p, to_01=False, normalize=False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if to_01:
        torchvision.utils.save_image((x+1)/2, p, normalize=normalize)
    else:
        torchvision.utils.save_image(x, p, normalize=normalize)


def mp(p):
    # if p is like a/b/c/d.png, then only make a/b/c/
    first_dot = p.find('.')
    last_slash = p.rfind('/')
    if first_dot < last_slash:
        assert ValueError('Input path seems like a/b.c/d/g, which is not allowed :(')
    p_new = p[:last_slash] + '/'
    if not os.path.exists(p_new):
        os.makedirs(p_new)


def get_plt_color_list():
    return ['red', 'green', 'blue', 'black', 'orange', 'yellow', 'black']


    
   
def draw_bound(a, m, color):
    if a.device != 'cpu':
        a = a.cpu()
    if color == 'red':
        c = torch.ones((3, 224, 224)) * torch.tensor([1, 0, 0])[:, None, None]
    if color == 'green':
        c = torch.ones((3, 224, 224)) * torch.tensor([0, 1, 0])[:, None, None]
    
    return c * m + a * (1 - m)

# class EasyDict(dict):
#     """Convenience class that behaves like a dict but allows access with the attribute syntax."""

#     def __getattr__(self, name: str) -> Any:
#         try:
#             return self[name]
#         except KeyError:
#             raise AttributeError(name)

#     def __setattr__(self, name: str, value: Any) -> None:
#         self[name] = value

#     def __delattr__(self, name: str) -> None:
#         del self[name]


def smooth_loss(output, weight):
    tv_loss = torch.sum(
        (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
        (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight



def compose_images_in_folder(p, dim, size=224):
    l = glob.glob(p + '*.png')
    l += glob.glob(p + '*.jpg')
    print(l)
    return torch.cat([load_png(item, size) for item in l], dim)



def get_bkg(m, e=0.01):
    assert  len(m.shape) == 4
    b = [0.2667, 0, 0.3255]
    m_0 = (m[:, 0, ...] > b[0] - e) * (m[:, 0, ...] < b[0] + e)
    m_1 = (m[:, 1, ...] > b[1] - e) * (m[:, 1, ...] < b[1] + e)
    m_2 = (m[:, 2, ...] > b[2] - e) * (m[:, 2, ...] < b[2] + e)
    m =   1. - (m_0 * m_1 * m_2).float()
    return m[None, ...]



def lpips_(a, b ):
    import lpips
    lpips_score = lpips.LPIPS(net='alex').to(a.device)
    return lpips_score(a, b)


def image_align(a, b):
    
    pass



def ssim_(p1, p2):
    i1 = cv2.imread(p1)
    i2 = cv2.imread(p2)
    
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    
    return ssim(i1, i2)

from math import log10, sqrt

def psnr_(a, b):
    original = cv2.imread(a)
    compressed = cv2.imread(b)
    
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
    
    


def psnr():
    pass