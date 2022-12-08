from torch import nn
import torch as th
from torch import clamp
from advfussion.script_util import create_model_and_diffusion
from advfussion.grad_cam import GradCamPlusPlus, get_last_conv_name

'''

'''
class AdvFussionAttack(nn.Module):
    def __init__(self, 
        model, 
        f_model_path, 
        adver_scale=0.4, 
        start_t=100, 
        range_t2_s=0, 
        range_t2_e=200, 
        mask_p=1, 
        nb_iter_conf=25, 
        early_stop=True, 
        use_fp16=True, 
        device='cuda', 
        timestep_respacing=250, 
        **kwargs):
        self.device = th.device(device)
        d = {'image_size':256, 
        'num_channels':256, 
        'num_res_blocks':2, 
        'num_heads':4, 
        'num_heads_upsample':-1, 
        'num_head_channels':64, 
        'attention_resolutions':"32,16,8", 
        'channel_mult':"", 
        'dropout':0.0, 
        'class_cond':True, 
        'use_checkpoint':False, 
        'use_scale_shift_norm':True, 
        'resblock_updown':True, 
        'use_fp16':use_fp16, 
        'use_new_attention_order':False, 
        'learn_sigma':True, 
        'diffusion_steps':1000, 
        'noise_schedule':"linear", 
        'timestep_respacing':[timestep_respacing], 
        'use_kl':False,
        'predict_xstart':False,
        'rescale_timesteps':False,
        'rescale_learned_sigmas':False,
        }
        super().__init__()
        self.f_model, self.diffusion = create_model_and_diffusion(**d)
        self.f_model.load_state_dict(th.load(f_model_path, map_location='cpu'))
        self.f_model.to(self.device)
        if use_fp16:  self.f_model.convert_to_fp16()
        self.f_model.eval()

        self.model = model
        layer_name = get_last_conv_name(model)
        self.grad_cam = GradCamPlusPlus(model, layer_name)

        self.start_t = start_t
        self.range_t2_s = range_t2_s
        self.range_t2_e = range_t2_e
        self.mask_p = mask_p
        self.adver_scale = adver_scale
        self.nb_iter_conf = nb_iter_conf
        self.early_stop = early_stop

    def cond_fn(self, x, t, y=None, mean=None, log_variance=None,  pred_xstart=None, 
            mask=0, **kwargs):
        time = int(t[0].detach().cpu())
        if  self.range_t2_s <=time <= self.range_t2_e:
            mask = mask.detach().clone() if mask!=0 else 0
            eps = th.exp(0.5 * log_variance)
            delta = th.zeros_like(x)
            with th.enable_grad():
                delta.requires_grad_()
                for _ in range(self.nb_iter_conf):
                    tmpx = pred_xstart.detach().clone() + delta  # range from -1~1
                    attack_logits = self.model(th.clamp((tmpx+1)/2.,0.,1.)) 
                    if self.early_stop:
                        target = y
                        sign = th.where(attack_logits.argmax(dim=1)==y, 1, 0)
                    else:
                        target = th.where(attack_logits.argmax(dim=1)==y, y, attack_logits.argmin(dim=1))
                        sign = th.where(attack_logits.argmax(dim=1)==y, 1, -1)
                    selected = sign * attack_logits[range(len(attack_logits)), target.view(-1)] 
                    loss = -selected.sum()
                    loss.backward()
                    grad_ = delta.grad.data.detach().clone()
                    delta.data += grad_  * self.adver_scale  *(1-mask)**self.mask_p
                    delta.data = clamp(delta.data, -eps, eps)
                    delta.grad.data.zero_()
            mean = mean.float() + delta.data.float() 
        return mean

    def forward(self, inputs, labels=None, use_cam=True, use_half=True):
        assert labels is not None
        if not use_half or inputs is None:
            labels = th.tensor(labels).to(self.device)
            sample = self.diffusion.p_sample_loop(
                lambda x, t, y, **kwargs: 
                    self.f_model(x, t, y),
                [len(labels), 3, 256, 256],
                clip_denoised=True,
                model_kwargs={"y":labels},
                cond_fn=self.cond_fn,
                device=self.device,
            )
        else:
            img, labels = inputs.to(self.device), labels.to(self.device)
            mask = self.grad_cam(img).unsqueeze(1) if use_cam else None
            x = img.clone().detach()*2-1
            sample = self.diffusion.p_sample_loop(
                lambda x, t, y, **kwargs: 
                    self.f_model(x, t, y),
                list(img.shape),
                clip_denoised=True,
                model_kwargs={"guide_x":x, 
                    "y":labels, "mask":mask,},
                cond_fn=self.cond_fn,
                start_t=self.start_t,
                device=self.device,
            )
        sample = th.clamp(sample,-1.,1.)
        return (sample+1)/2
    
    def __del__(self):
        self.grad_cam.remove_handlers()
        del self.f_model, self.diffusion

if __name__ == '__main__':
    from robustbench.utils import load_model
    from  torchvision import utils as vutils
    model = load_model(model_name='Standard_R50', 
        dataset='imagenet', threat_model='Linf')
    model = model.cuda().eval()
    model_path = '/root/256x256_diffusion.pt'
    attacker =  AdvFussionAttack(model, model_path)
    sample = attacker(None, labels=[1,2,3])
    vutils.save_image(sample, './test.jpg', normalize=True)
