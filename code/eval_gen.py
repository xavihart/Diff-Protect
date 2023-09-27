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

from clip_similarity import clip_sim





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
        
        subimage_path = get_dir_name_from_config(mode, g_mode, using_target, target_rate=target_rate, prefix='out_fid')
        save_path = get_dir_name_from_config(mode, g_mode, using_target, target_rate=target_rate, prefix='out_genscore')
        
        mp(save_path)
        
        t_list = [0.01, 0.05, 0.1, 0.2, 0.3]
        
        
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

            clip_score_list = {
                
            }
            psnr_score_list = {

            }
            lpips_score_list = {

            }
            ssim_score_list = {

            }

            for t_ in t_list:
                clip_score_list[t_] = []
                psnr_score_list[t_] = []
                lpips_score_list[t_] = []
                ssim_score_list[t_] = []
            
            for i in tqdm(range(100)):
                img_path = os.path.join(dir_name, style, f'{i}_attacked.png')
                clean_img_path = os.path.join(clean_name, style, f'{i}_attacked.png')
                
                gen_img_path = os.path.join(dir_name, style, f'{i}_multistep.png')
                
                if not os.path.exists(img_path):
                    print("NO SUCH PATH", os.path.join(dir_name, style, f'{i}_attacked.png'))
                    break
                
                # x_adv = load_png(img_path, 512)[None, ...].to(device)
                # x     = load_png(clean_img_path, 512)[None, ...].to(device)
                # x_gen = load_png(gen_img_path, None)[None, ...].to(device) # 1 * 3 * 224 * (6:224)
                
                

                for j, t in enumerate(t_list):
                    # x_gen_sub = x_gen[:, :, :, (j+1)*512:(j+1)*512+512]
                    
                    save_p = subimage_path + f'{style}/{t}/'
                    # mp(save_p)
                    # # print(save_p + f'{i}.png')
                    # si(x_gen_sub, save_p + f'{i}.png')

                    x_gen_sub = load_png(save_p + f'{i}.png', 512)[None, ...].to(device)

                    # print(x_gen_sub.shape)

                    clip_score = clip_sim(save_p + f'{i}.png', clean_img_path)

                    ssim_x = ssim_(save_p + f'{i}.png', clean_img_path)
                    # lpips_x = lpips_(x, x_adv)
                    psnr_x = psnr_(save_p + f'{i}.png', clean_img_path)

                    lpips_score = lpips_(load_png(save_p + f'{i}.png', 512),load_png(clean_img_path, 512))
                    lpips_score = lpips_score[0, 0, 0, 0].cpu().tolist()

                    clip_score_list[t].append(clip_score)
                    psnr_score_list[t].append(psnr_x)
                    lpips_score_list[t].append(lpips_score)
                    ssim_score_list[t].append(ssim_x)

                        
                    
                
                
            
            
            save_path_style = os.path.join(save_path, style+'/')
            mp(save_path_style)

            torch.save({
                'clip_score':clip_score_list,
                'psnr_score':psnr_score_list,
                'lpips_score':lpips_score_list,
                'ssim_score': ssim_score_list

            }, save_path_style+'/metrics.bin')
            
            
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