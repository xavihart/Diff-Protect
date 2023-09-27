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
from utils import mp, si, cprint

from sklearn.decomposition import PCA







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


class identity_loss(nn.Module):
    """
    An identity loss used for input fn for advertorch. To support semantic loss,
    the computation of the loss is implemented in class targe_model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class target_model(nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model,
                 condition: str,
                 target_info: str = None,
                 mode: int = 2, 
                 rate: int = 10000, g_mode='+'):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss.
        :param target_info: The target textural for textural loss.
        :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused
        :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.fn = nn.MSELoss(reduction="sum")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate
        self.g_mode = g_mode
        
        print('g_mode:',  g_mode)

    def get_components(self, x):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """

        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(device)
        c = self.model.get_learned_conditioning(self.condition)
        loss = self.model(z, c)[0]
        return z, loss

    def forward(self, x, components=False):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """
        g_dir = 1. if self.g_mode == '+' else -1.
        zx, loss_semantic = self.get_components(x)
        zy, loss_semantic_y = self.get_components(self.target_info)
        if components:
            return self.fn(zx, zy), loss_semantic
        if self.mode == 'advdm': # 
            return - loss_semantic * g_dir
        elif self.mode == 'texture_only':
            return self.fn(zx, zy)
        elif self.mode == 'mist':
            return self.fn(zx, zy) * g_dir  - loss_semantic * self.rate
        else:
            raise KeyError('mode not defined')


def init(epsilon: int = 16, steps: int = 100, alpha: int = 1, 
         input_size: int = 512, object: bool = False, seed: int =23, 
         ckpt: str = None, base: str = None, mode: int = 2, rate: int = 10000, g_mode='+'):
    """
    Prepare the config and the model used for generating adversarial examples.
    :param epsilon: Strength of adversarial attack in l_{\infinity}.
                    After the round and the clip process during adversarial attack, 
                    the final perturbation budget will be (epsilon+1)/255.
    :param steps: Iterations of the attack.
    :param alpha: strength of the attack for each step. Measured in l_{\infinity}.
    :param input_size: Size of the input image.
    :param object: Set True if the targeted images describes a specifc object instead of a style.
    :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused. 
                 See the document for more details about the mode.
    :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
    :returns: a dictionary containing model and config.
    """

    if ckpt is None:
        ckpt = 'ckpt/model.ckpt'

    if base is None:
        base = 'configs/stable-diffusion/v1-inference-attack.yaml'

    # seed_everything(seed)
    imagenet_templates_small_style = ['a painting']
    imagenet_templates_small_object = ['a photo']

    config_path = os.path.join(os.getcwd(), base)
    config = OmegaConf.load(config_path)

    ckpt_path = os.path.join(os.getcwd(), ckpt)
    model = load_model_from_config(config, ckpt_path).to(device)

    fn = identity_loss()

    if object:
        imagenet_templates_small = imagenet_templates_small_object
    else:
        imagenet_templates_small = imagenet_templates_small_style

    input_prompt = [imagenet_templates_small[0] for i in range(1)]
    net = target_model(model, input_prompt, mode=mode, rate=rate, g_mode=g_mode)
    net.eval()

    # parameter
    parameters = {
        'epsilon': epsilon/255.0 * (1-(-1)),
        'alpha': alpha/255.0 * (1-(-1)),
        'steps': steps,
        'input_size': input_size,
        'mode': mode,
        'rate': rate,
        'g_mode': g_mode
    }

    return {'net': net, 'fn': fn, 'parameters': parameters}


def infer(img: PIL.Image.Image, config, tar_img: PIL.Image.Image = None, diff_pgd=None, using_target=False) -> np.ndarray:
    """
    Process the input image and generate the misted image.
    :param img: The input image or the image block to be misted.
    :param config: config for the attack.
    :param img: The target image or the target block as the reference for the textural loss.
    :returns: A misted image.
    """

    net = config['net']
    fn = config['fn']
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters["epsilon"]
    g_mode = parameters['g_mode']
    
    cprint(f'epsilon: {epsilon}', 'y')
    
    alpha = parameters["alpha"]
    steps = parameters["steps"]
    input_size = parameters["input_size"]
    rate = parameters["rate"]

    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img[:, :, :3]
    if tar_img is not None:
        tar_img = np.array(tar_img).astype(np.float32) / 127.5 - 1.0
        tar_img = tar_img[:, :, :3]
    trans = transforms.Compose([transforms.ToTensor()])
    
    data_source = torch.zeros([1, 3, input_size, input_size]).to(device)
    data_source[0] = trans(img).to(device)
    target_info = torch.zeros([1, 3, input_size, input_size]).to(device)
    target_info[0] = trans(tar_img).to(device)
    net.target_info = target_info
    net.mode = mode
    net.rate = rate
    label = torch.zeros(data_source.shape).to(device)
    print(net(data_source, components=True))

    # Targeted PGD attack is applied.
    
    if mode in ['advdm', 'texture_only', 'mist',]: # using raw PGD
            
        attack = LinfPGDAttack(net, fn, epsilon, steps, eps_iter=alpha, clip_min=-1.0, targeted=True)
        attack_output = attack.perturb(data_source, label)
    
    elif mode == 'sds': # apply SDS to speed up the PGD
        
        print('using sds')
        
        attack =  Linf_PGD(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        
        attack_output, loss_all = attack.pgd_sds(X=data_source, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_image=target_info, target_rate=rate)
        
        print(torch.abs(attack_output-data_source).max())
        print(loss_all)
        # emp = [wandb.log({'adv_loss':loss_item}) for loss_item in loss_all]
    
    editor = SDEdit(net=net)
    edit_multi_step = editor.edit_list(attack_output, restep=None)
    edit_one_step   = editor.edit_list(attack_output, restep='ddim100')
        
    
    print(net(attack_output, components=True))

    output = attack_output[0]
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    grid_adv = grid_adv
    return grid_adv, edit_multi_step, edit_one_step


# Test the script with command: python mist_v2.py 16 100 512 1 2 1
# or the command: python mist_v2.py 16 100 512 2 2 1, which process
# the image blockwisely for lower VRAM cost


def normalized_(x):
    return (x - x.min() / x.max() - x.min())

def main():

    x_path = '/ssdscratch/hxue45/data/phd_2/diff_mist/test_images/portrait/trevor_4.jpg'
    x_adv_path = '/ssdscratch/hxue45/data/phd_2/diff_mist/out/sds_eps16_steps100_gmode-/portrait/trevor_4_attacked.png'
    x_adv_path = '/ssdscratch/hxue45/data/phd_2/diff_mist/out/texture_only_eps16_steps100_gmode+/portrait/trevor_4_attacked.png'
    x_adv_path = '/ssdscratch/hxue45/data/phd_2/diff_mist/out/advdm_eps16_steps100_gmode+/portrait/trevor_4_attacked.png'
    
    
    x_path = '/ssdscratch/hxue45/data/phd_2/diff_mist/out_iclr/none_eps16_steps100_gmode-/artwork/1_attacked.png'
    x_adv_path = '/ssdscratch/hxue45/data/phd_2/diff_mist/out_iclr/texture_only_eps16_steps100_gmode+/artwork/1_attacked.png'
    
    x_path = '/ssdscratch/hxue45/data/phd_2/diff_mist/test_images/portrait/elon_2.jpg'
    x_adv_path = '/ssdscratch/hxue45/data/phd_2/diff_mist/out/sds_z_eps100_steps100_gmode+/portrait/elon_2_attacked.png'
    device = 0
    
    ckpt = 'ckpt/model.ckpt'
    base = 'configs/stable-diffusion/v1-inference-attack.yaml'

    config_path = os.path.join(os.getcwd(), base)
    config = OmegaConf.load(config_path)

    ckpt_path = os.path.join(os.getcwd(), ckpt)
    model = load_model_from_config(config, ckpt_path).to(device)

    from utils import load_png
    x = load_png(x_path, 512)[None, ...].to(device)
    x_adv = load_png(x_adv_path, 512)[None, ...].to(device)

    x = x * 2 - 1
    x_adv = x_adv * 2 - 1

    print(torch.abs(x-x_adv).max() * 255)
    print("l1", torch.abs(x-x_adv).norm(p=1))

    z = model.get_first_stage_encoding(model.encode_first_stage(x)).to(device)
    z_adv = model.get_first_stage_encoding(model.encode_first_stage(x_adv)).to(device)
    

    
    x_decode = model.decode_first_stage(z, force_not_quantize=True)
    x_adv_decode = model.decode_first_stage(z_adv, force_not_quantize=True)

    x_decode = (x_decode + 1) / 2
    x_adv_decode = (x_adv_decode + 1) / 2
    
    print(torch.abs(z-z_adv).max() * 255)
    print("l1", torch.abs(z-z_adv).norm(p=1))

    print(torch.abs(x_decode-x_adv_decode).max() * 255)
    

    


    from utils import si
    si(torch.cat([torch.cat([(x+1)/2, x_decode], -1), torch.cat([(x_adv+1)/2, x_adv_decode], -1)], -2), 'demo.png')
    si(torch.cat([z[:, :3], z_adv[:, :3]]), 'demo_mean.png')
    si(torch.cat([z[:, [-1]], z_adv[:, [-1]]]), 'demo_var.png')
    
    si(torch.cat([normalized_(z[:, :3]), normalized_(z_adv[:, :3])]), 'demo_mean_normed.png')
    si(torch.cat([normalized_(z[:, [-1]]), normalized_(z_adv[:, [-1]])]), 'demo_var_normed.png')




    
    

if __name__ == '__main__':
    main()