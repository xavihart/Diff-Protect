
<div align="center">

<h2>Toward effective protection against diffusion-based mimicry through score distillation [ICLR'2024]</h2>

**[Haotian Xue <sup>1](https://xavihart.github.io/), [Chumeng Liang <sup>2,3](https://caradryanliang.github.io/), [Xiaoyu Wu <sup>2](https://openreview.net/profile?id=~Xiaoyu_Wu1), and [Yongxin Chen <sup>1](https://yongxin.ae.gatech.edu/)**


<sup>1</sup> Georgia Tech <sup>2</sup> SJTU <sup>3</sup> USC


</div>




![](test_images/media/teaser.png)

TL;DR : novel insights into attacks agasint LDM, a more effective protection against malicious diffusion model editing


## Updates
- [01/15/2024] 🎉 Our paper is accepted to ICLR'2024!
- [11/24/2023] Paper finally shown on [Arxiv](https://arxiv.org/abs/2311.12832)
- [09/27/2023] Paper will be released soon!
- [09/27/2023] Our repo is alive!








## Quick Setup

Install the envs:

```
conda env create -f env.yml
conda activate mist
pip install --force-reinstall pillow
```

Download the checkpoint of LDM [Stable-diffusion-model v1.4 checkpoint]

```
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir ckpt
mv sd-v1-4.ckpt ckpt/model.ckpt
```


## Run the Protection 

Configs can be set in `configs/attack/base.yaml`, some keys arguments include:

```
attack:
    epsilon: 16 # l_inf budget
    steps: 100 # attack steps
    input_size: 512 # image size
    mode: sds 
    img_path: [IMAGE_DIR_PATH]
    output_path: [OUT_PATH]
    alpha: 1 # step size
    g_mode: "+"
    use_wandb: False
    project_name: iclr_mist
    diff_pgd: [False, 0.2, 'ddim100'] # un-used feature
    using_target: False
    target_rate: 5 # scale factor for L_S and L_T
    device: 0 # GPU id

```


here we offer some scripts to run different type of protections:

AdvDM:
```
python code/diff_mist.py attack.mode='advdm' attack.g_mode='+'
```
PhotoGuard:
```
python code/diff_mist.py attack.mode='texture_only' attack.g_mode='+'
```

Mist:
```
python code/diff_mist.py attack.mode='mist' attack.g_mode='+'
```
AdvDM(-):
```
python code/diff_mist.py attack.mode='advdm' attack.g_mode='-'
```
SDS(+):
```
python code/diff_mist.py attack.mode='sds' attack.g_mode='+'
```
SDS(-):
```
python code/diff_mist.py attack.mode='sds' attack.g_mode='-'
```
SDST(-):
```
python code/diff_mist.py attack.mode='sds' attack.g_mode='-' attack.using_target=True
```

the output includes: `[NAME]_attacked.png` which is the attacked image, `[NAME]_multistep.png` which is the SDEdit results, and `[NAME]_onestep.png` which is the onestep x_0 prediction results.





<img src="out/advdm_eps16_steps100_gmode+/to_protect/suzume_attacked.png" alt="drawing" width="200"/>  <img src="out/mist_eps16_steps100_gmode+/to_protect/suzume_attacked.png" alt="drawing" width="200"/> <img src="out/sds_eps16_steps100_gmode-/to_protect/suzume_attacked.png" alt="drawing" width="200"/>

[From let to right]: AdvDM, Mist, SDS(-), using eps=16, SDS-version is much more effective than the previous two methods


## Cited as:

```
@inproceedings{xue2023toward,
  title={Toward effective protection against diffusion-based mimicry through score distillation},
  author={Xue, Haotian and Liang, Chumeng and Wu, Xiaoyu and Chen, Yongxin},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```