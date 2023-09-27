# @ Haotian Xue 2023
# accelerated version: mist v3
# feature 1: SDS
# feature 2: Diff-PGD

import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from advertorch.attacks import LinfPGDAttack
from attacks import Linf_PGD, SDEdit
import time
import wandb
import glob
import hydra
from utils import mp, si, cprint, load_png
from sklearn.decomposition import PCA
import matplotlib.pylab as plt

from utils import lpips_, ssim_, psnr_






ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'




def load_image_from_path(image_path: str, input_size: int) -> PIL.Image.Image:
    """
    Load image form the path and reshape in the input size.
    :param image_path: Path of the input image
    :param input_size: The requested size in int.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    img = Image.open(image_path).resize((input_size, input_size),
                                        resample=PIL.Image.BICUBIC)
    return img


def load_model_from_config(config, ckpt, verbose: bool = False):
    """
    Load model from the config and the ckpt path.
    :param config: Path of the config of the SDM model.
    :param ckpt: Path of the weight of the SDM model
    :param verbose: Whether to show the unused parameters weight.
    :returns: A SDM model.
    """
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # Support loading weight from NovelAI
    if "state_dict" in sd:
        import copy
        sd_copy = copy.deepcopy(sd)
        for key in sd.keys():
            if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                sd_copy[newkey] = sd[key]
                del sd_copy[key]
        sd = sd_copy

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model




def normalized_(x):
    return (x - x.min() / x.max() - x.min())


def get_dir_name_from_config(mode, g_mode, using_target, eps=16, steps=100, target_rate=5, prefix='out_iclr'):
    if using_target and mode == 'sds':
        mode_name = f'sdsT{target_rate}'
    else:
        mode_name = mode
    dir_name = f'{prefix}/{mode_name}_eps{eps}_steps{steps}_gmode{g_mode}/'
    return dir_name


EXP_LIST = [
    ('advdm', '+', False, -1),
    ('advdm', '-', False, -1),
    ('mist', '+', False, -1),
    ('sds', '+', False, -1),
    ('sds', '-', False, -1),
    ('sds', '-', True, 1),
    ('sds', '-', True, 5),
    ('texture_only', '+', False, -1),
    ('none', '-', False, -1),

]

STYLE_LIST = [
    'anime',
    'artwork',
    'landscape',
    'portrait'
]

def dm_range(x):
    return 2 * x - 1

def rdm_range(x):
    return (x+1)/2

def encode(x, model):
    z = model.get_first_stage_encoding(model.encode_first_stage(x)).to(x.device)
    return z

def linf(x):
    return torch.abs(x).max()

@torch.no_grad()
def main():
    
    
    # ckpt = 'ckpt/model.ckpt'
    # base = 'configs/stable-diffusion/v1-inference-attack.yaml'

    # config_path = os.path.join(os.getcwd(), base)
    # config = OmegaConf.load(config_path)

    # ckpt_path = os.path.join(os.getcwd(), ckpt)
    # model = load_model_from_config(config, ckpt_path).to(device)
    
    
    
    
    for exp_config in tqdm(EXP_LIST):
        cprint(exp_config, 'y')
        mode, g_mode, using_target, target_rate = exp_config
        
        dir_name = get_dir_name_from_config(mode, g_mode, using_target, target_rate=target_rate)
        cprint('fetching dir: ' + dir_name, 'g')
        
        clean_name = get_dir_name_from_config('none', '-', using_target=False, target_rate=target_rate)
        
        
        save_path = get_dir_name_from_config(mode, g_mode, using_target, target_rate=target_rate, prefix='out_stat')
        
        
        mp(save_path)
        
        
        for style in STYLE_LIST:
            print(style)
            
            linf_z_list = []
            linf_x_list = []
            
            ssim_list = []
            lpips_list = []
            psnr_list = []
            

            
            z_max = 0
            
            x_list = []
            x_adv_list =[]
            
            for i in tqdm(range(100)):
                img_path = os.path.join(dir_name, style, f'{i}_attacked.png')
                clean_img_path = os.path.join(clean_name, style, f'{i}_attacked.png')
                
                if not os.path.exists(img_path):
                    print("NO SUCH PATH", os.path.join(dir_name, style, f'{i}_attacked.png'))
                    break
            
                x_adv = load_png(img_path, 512)[None, ...].to(device)
                x     = load_png(clean_img_path, 512)[None, ...].to(device)
                # print(x_adv.shape, x.shape)
                
                # z-space
                x, x_adv = dm_range(x), dm_range(x_adv)
                
                x_list.append(x)
                x_adv_list.append(x_adv)
                
                ssim_x = ssim_(img_path, clean_img_path)
                # lpips_x = lpips_(x, x_adv)
                psnr_x = psnr_(img_path, clean_img_path)
                
                # print(ssim_x, lpips_x, psnr_x)
                
                ssim_list.append(ssim_x)
                # lpips_list.append(lpips_x)
                psnr_list.append(psnr_x)
                
                # z, z_adv = encode(x, model), encode(x_adv, model)
                
                # x, x_adv = rdm_range(x), rdm_range(x_adv)
                
                
                # si(torch.cat([x, x_adv], -1), os.path.join(save_path, style+'/', f'x_{i}.png'))
                # si(torch.cat([z[:, :3], z_adv[:, :3]], -1), os.path.join(save_path, style+'/', f'z_{i}.png'))
                
                # norm_max = z.max()
                # norm_min = z.min()
                
                # z, z_adv = (z - norm_min)/(norm_max-norm_min), (z_adv - norm_min)/(norm_max-norm_min)
                
                # if z.abs().max() > z_max:
                #     z_max = z.abs().max()
                
                
                # linf_x = linf(x-x_adv)
                # linf_z = linf(z-z_adv)
                
                # print(linf_x , linf_z)
                
                # linf_z_list.append(linf_z.item())
                # linf_x_list.append(linf_x.item())
            
            save_path_style = os.path.join(save_path, style+'/')
            mp(save_path_style)
            # torch.save({
            #     'linf_x':linf_x_list,
            #     'linf_z':linf_z_list
            # }, save_path_style+'/linf.bin')
            
            x_adv_all = torch.cat(x_adv_list, 0)
            x_all = torch.cat(x_list, 0)
            lpips_score = lpips_(x_all, x_adv_all)
            lpips_score = lpips_score[:, 0, 0, 0].cpu().tolist()
            
            torch.save({
                'ssim':ssim_list,
                'lpips':lpips_score,
                'psnr':psnr_list
            }, save_path_style+'/x_adv_metrics.bin')
            
            
            # plt.plot()
                
                
                
                
                
                
                
                
                
                
                
                
            

    # from utils import load_png
    # x = load_png(x_path, 512)[None, ...].to(device)
    # x_adv = load_png(x_adv_path, 512)[None, ...].to(device)

    # x = x * 2 - 1
    # x_adv = x_adv * 2 - 1

    # print(torch.abs(x-x_adv).max() * 255)
    # print("l1", torch.abs(x-x_adv).norm(p=1))

    # z = model.get_first_stage_encoding(model.encode_first_stage(x)).to(device)
    # z_adv = model.get_first_stage_encoding(model.encode_first_stage(x_adv)).to(device)
    

    
    # x_decode = model.decode_first_stage(z, force_not_quantize=True)
    # x_adv_decode = model.decode_first_stage(z_adv, force_not_quantize=True)
    
    

    
    

if __name__ == '__main__':
    main()