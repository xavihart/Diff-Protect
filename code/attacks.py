import torch
import numpy as np
from tqdm import tqdm
import torchvision
from ldm.models.diffusion.ddim import DDIMSampler
import wandb



ATTACK_TYPE = [
    'PGD',
    'PGD_SDS',
    'Diff_PGD',
    'Diff_PGD_SDS'
]

class SDEdit():
    def __init__(self, net):
        self.dm = net.model # SD model
        self.c = net.condition
        
    def edit_list(self, x, restep=None, t_list=[0.01, 0.05, 0.1, 0.2, 0.3]):
        l = [x.cpu()]
        for t in t_list:
            x_return = self.edit(x, t, restep=restep)
            l.append(x_return)
        return (torch.cat(l, dim=-1) + 1 ) /2

    @torch.no_grad()
    def edit(self, x, guidance, restep=None, from_z=False, c=None):
        # create DDIM Sampler
        # guidance from 0-1
        assert guidance >= 0  and guidance < 1
        
        if c is None:
            cond_prompt = self.c
        else:
            cond_prompt = c
        
        
        if restep is None:
            ##
            # directly use one step denoise
            ##
            t_guidance = int(self.dm.num_timesteps * guidance)
            # latent z
            z = self.dm.get_first_stage_encoding(self.dm.encode_first_stage(x)).to(x.device) if not from_z else x
            t = torch.full((z.shape[0],), t_guidance, device=z.device, dtype=torch.long)
                            
            # sample noise
            noise = torch.randn_like(z)
            z_noisy = self.dm.q_sample(x_start=z, t=t, noise=noise)
            
            # get z_t
            cnd = self.dm.get_learned_conditioning(cond_prompt)
            eps_pred = self.dm.apply_model(z_noisy, t, cond=cnd) # \hat{eps}

            
            # get \hat{x_0}
            
            z_0_pred = self.dm.predict_start_from_noise(z_noisy, t, eps_pred)
            if guidance == 0:
                z_0_pred = z
            x_0_pred = self.dm.decode_first_stage(z_0_pred, force_not_quantize=True)
            return x_0_pred.cpu()   
                     
        else:
            # ddim
            ddim_steps = int(restep[4:])
            t = int(ddim_steps * guidance)
            
            sampler = DDIMSampler(self.dm, schedule="linear")
            sampler.make_schedule(ddim_steps, 'uniform', verbose=False)
            
            # sdedit-add-noise (q-sample)
            real_starting_t = sampler.ddim_timesteps[t]
            z = self.dm.get_first_stage_encoding(self.dm.encode_first_stage(x)).to(x.device) if not from_z else x
            t_origin_model = torch.full((z.shape[0],), real_starting_t, device=z.device, dtype=torch.long)
            noise = torch.randn_like(z)
            z_t = self.dm.q_sample(x_start=z, t=t_origin_model, noise=noise)
            
            cnd = self.dm.get_learned_conditioning(cond_prompt)
            
            # denoise for t steps:
            for i, step in enumerate(reversed(sampler.ddim_timesteps[:t])):
                index = t - i  - 1
                ts = torch.full((z.shape[0],), step, device=x.device, dtype=torch.long)
                z_t, z_pred_0 = sampler.p_sample_ddim(z_t, cnd, ts, index=index)
            
            x_return = self.dm.decode_first_stage(z_t, force_not_quantize=True)
            return x_return.cpu()
    


class Linf_PGD():
    def __init__(self, net, fn, epsilon, steps, eps_iter, clip_min, clip_max=1, targeted=True, attack_type='PGD', g_mode="+", diff_pgd=None):
        self.net = net
        self.fn = fn
        self.eps = epsilon
        self.step_size = eps_iter
        self.iters = steps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        
        self.attack_type = attack_type
        self.g_mode = g_mode
        self.g_dir = 1. if g_mode == '+' else -1.
        self.diff_pgd = diff_pgd

        self.cirt = torch.nn.MSELoss(reduction="sum")
        
        
        if self.attack_type not in ATTACK_TYPE:
            raise AttributeError(f"self.attack_type should be in {ATTACK_TYPE}, \
                                 {self.attack_type} is an undefined")
        
    # interface to call all the attacks
    def perturbe(self, X, y):
        if self.attack_type == 'PGD':
            return self.pgd(X, y)
        elif self.attack_type == 'PGD_SDS':
            return self.pgd_sds(X, y)
    
    # traditional pgd
    def pgd(self, X, y):
        
        # add uniform random start [-eps, eps]
        X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*self.eps-self.eps).cuda()
        
        pbar = tqdm(range(self.iters))
        
        # modified from photoguard by Salman et al.
        for i in pbar:
            actual_step_size = self.step_size - (self.step_size - self.step_size / 100) / self.iters * i  

            X_adv.requires_grad_(True)

            #loss = (model(X_adv).latent_dist.mean).norm()
            
            loss = self.fn(self.net(X_adv), y)

            pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

            grad, = torch.autograd.grad(loss, [X_adv])
            
            # update
            X_adv = X_adv - grad.detach().sign() * actual_step_size
            # clip
            X_adv = torch.minimum(torch.maximum(X_adv, X - self.eps), X + self.eps)
            X_adv.data = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)
            X_adv.grad = None    
            
                
        return X_adv
    
    @staticmethod
    def mp(p):
        import os
        # if p is like a/b/c/d.png, then only make a/b/c/
        first_dot = p.find('.')
        last_slash = p.rfind('/')
        if first_dot < last_slash:
            assert ValueError('Input path seems like a/b.c/d/g, which is not allowed :(')
        p_new = p[:last_slash] + '/'
        if not os.path.exists(p_new):
            os.makedirs(p_new)

    
    def pgd_sds(self, X, net, c, label=None, random_start=False, delta_score=False, name='test', diff_pgd=False, using_target=False, target_image=None, target_rate=None):
        
        diff_pgd, diff_pgd_t, diff_pgd_respace = diff_pgd
        
        save_path = f'test_sds/{name}_eps{self.eps}_iter{self.iters}_gmode{(self.g_mode)}/'
        

        self.mp(save_path)
        
        editor = SDEdit(net)
        
        
        # gradient required from SDS, torch.no_grad() here
        if random_start:
            print("using random_start")
            X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*self.eps-self.eps).cuda()
        else:
            X_adv = X.clone().detach()
        
        # pbar = tqdm(range(self.iters))
        pbar = range(self.iters)
        
        dm = net.model # SD model to call
        
        loss_all = []
        
        for ppp in  [0.99]:
            for i in pbar:
                
                
                with torch.no_grad():
                    if diff_pgd:
                        X_adv_raw = X_adv.clone().detach()
                        X_adv = editor.edit(X_adv, diff_pgd_t, diff_pgd_respace).clone().detach().to(X.device)
                    else:
                        X_adv_raw = X_adv.clone().detach()
                

                # actual_step_size = self.step_size - (self.step_size - self.step_size / 100) / self.iters * i 
                actual_step_size = self.step_size
                # SDS update, with only forward function
                
                X_adv.requires_grad_(True)
                z_adv = dm.get_first_stage_encoding(dm.encode_first_stage(X_adv)).to(X.device)
                
                z = z_adv.clone().detach()

                with torch.no_grad():
                    # to latent
                    
                    # sample noise
                    T = dm.num_timesteps
                    # t_to_set = int(T * ppp)
                    # t = torch.randint(t_to_set, t_to_set+1, (z.shape[0],), device=z.device).long()
                    t = torch.randint(0, T, (z.shape[0],), device=z.device).long()
                    # t = t * 0 + T - 10 * i -1
                    
                    
                    
                    # print(t)
                    noise = torch.randn_like(z)
                    
                    
                    # get z_t
                    z_noisy = dm.q_sample(x_start=z, t=t, noise=noise)
                    cnd = dm.get_learned_conditioning(c)
                    eps_pred = dm.apply_model(z_noisy, t, cond=cnd) # \hat{eps}
                    
                    # update z
                    grad = (eps_pred - noise)
                    
                        

                    loss = grad.norm(p=2).cpu().item()
                    # print(loss)
                    # loss_all.append(loss)
                
                torch.cuda.empty_cache()
                # get gradient wrt VAE (with gradient)
                # X_adv = X_adv.clone().detach()
                # X_adv.requires_grad_(True)
                # z = dm.get_first_stage_encoding(dm.encode_first_stage(X_adv)).to(X.device)
                z_adv.backward(gradient=grad)
                g_x = X_adv.grad.detach()

                if using_target:

                    with torch.no_grad():
                        z_target = dm.get_first_stage_encoding(dm.encode_first_stage(target_image)).to(X.device).detach()
                    
                    X_adv = X_adv.clone().detach()
                    X_adv.requires_grad_(True)
                    x_target = dm.get_first_stage_encoding(dm.encode_first_stage(X_adv)).to(X.device)    
                    loss = self.cirt(z_target, x_target)
                    loss.backward()
                    g_tex = X_adv.grad.detach()
                
                        
                
                # update x_adv
                # X_adv = X_adv - g_x.detach().sign() * actual_step_size
                if not using_target:
                    X_adv = X_adv_raw + self.g_dir * g_x.detach().sign() * actual_step_size 
                else:
                    X_adv = X_adv_raw + (self.g_dir * g_x.detach()* target_rate- g_tex).sign()  * actual_step_size
                X_adv = torch.minimum(torch.maximum(X_adv, X - self.eps), X + self.eps)
                X_adv.data = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)
                X_adv.grad=None
        
        with torch.no_grad():
            if diff_pgd:
                X_adv = editor.edit(X_adv, diff_pgd_t, diff_pgd_respace).clone().detach().to(X.device)
        


        return X_adv, loss_all
                    
                    
    
    # only update z
    def pgd_sds_latent(self, z, net, c, label=None, random_start=False, delta_score=False, name='test', diff_pgd=False, using_target=False, target_image=None, target_rate=None):
        
        diff_pgd, diff_pgd_t, diff_pgd_respace = diff_pgd
        
        save_path = f'test_sds/{name}_eps{self.eps}_iter{self.iters}_gmode{(self.g_mode)}/'
        

        self.mp(save_path)
        
        editor = SDEdit(net)
        
        z_raw = z.clone().detach()
        
        # gradient required from SDS, torch.no_grad() here
        if random_start:
            print("using random_start")
            z_adv = z.clone().detach() + (torch.rand(*X.shape)*2*self.eps-self.eps).cuda()
        else:
            z_adv = z.clone().detach()
        
        pbar = tqdm(range(self.iters))
        # z_raw = None
        
        dm = net.model # SD model to call
        
        loss_all = []
        
        for ppp in  [0.99]: # abandoned
            for i in pbar:
                
                
    
                

                # actual_step_size = self.step_size - (self.step_size - self.step_size / 100) / self.iters * i 
                actual_step_size = self.step_size
                # SDS update, with only forward function
                with torch.no_grad():
                    z = z_adv.clone().detach()
                    
                    # sample noise
                    T = dm.num_timesteps
                    # t_to_set = int(T * ppp)
                    # t = torch.randint(t_to_set, t_to_set+1, (z.shape[0],), device=z.device).long()
                    t = torch.randint(0, T, (z.shape[0],), device=z.device).long()
                    # t = t * 0 + T - 10 * i -1
                    
                    
                    
                    print(t)
                    noise = torch.randn_like(z)
                    
                    
                    # get z_t
                    z_noisy = dm.q_sample(x_start=z, t=t, noise=noise)
                    cnd = dm.get_learned_conditioning(c)
                    eps_pred = dm.apply_model(z_noisy, t, cond=cnd) # \hat{eps}
                    
                    # update z
                    grad = (eps_pred - noise)
                    
                        

                    loss = grad.norm(p=2).cpu().item()
                    # print(loss)
                    loss_all.append(loss)
                
                torch.cuda.empty_cache()

                # update x_adv
                # X_adv = X_adv - g_x.detach().sign() * actual_step_size
                z_adv = z_adv + self.g_dir * grad.detach().sign() * actual_step_size 
                
                z_adv = torch.minimum(torch.maximum(z_adv, z_raw - self.eps), z_raw + self.eps)
                
                z_adv.grad=None

        # decode into x space
        x_adv = dm.decode_first_stage(z_adv, force_not_quantize=True)


        return x_adv, loss_all
                    
                    
                    
                    
            
            
                    
            
        
