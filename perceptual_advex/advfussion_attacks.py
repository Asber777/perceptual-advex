from torch import nn
import torch as th
from torch import clamp
from advfussion.script_util import create_model_and_diffusion
from advfussion.grad_cam import GradCamPlusPlus, get_last_conv_name
from advfussion.free.ModelCondition import UNet
from advfussion.free.DiffusionCondition import GaussianDiffusionSampler

class AdvFussionAttack(nn.Module):
    def __init__(self, 
        model, 
        adver_scale, 
        start_t, 
        range_t2_s, 
        range_t2_e, 
        mask_p, 
        nb_iter_conf, 
        early_stop=True, 
        device='cuda',
        use_cam=True):
        super().__init__()
        self.model = model
        self.f_model = None
        self.diffusion = None
        self.grad_cam = None
        self.start_t = start_t
        self.range_t2_s = range_t2_s
        self.range_t2_e = range_t2_e
        self.mask_p = mask_p
        self.adver_scale = adver_scale
        self.nb_iter_conf = nb_iter_conf
        self.early_stop = early_stop
        self.device = th.device(device)
        self.use_cam = use_cam

    def __del__(self):
        if self.grad_cam is not None:
            self.grad_cam.remove_handlers()
        del self.f_model, self.diffusion

class AdvFussionImagNet(AdvFussionAttack):
    def __init__(self, 
        model, 
        f_model_path, 
        adver_scale=0.4, 
        start_t=50,  # 100
        range_t2_s=0, 
        range_t2_e=200, 
        mask_p=1, 
        nb_iter_conf=25, 
        early_stop=True, 
        device='cuda', 
        use_fp16=True, 
        timestep_respacing=250, 
        use_cam=True,
        **kwargs):
        
        self.d = {'image_size':256, 
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
        super().__init__(model, adver_scale, start_t, 
            range_t2_s, range_t2_e, mask_p, nb_iter_conf,
            early_stop, device, use_cam)
        self.f_model, self.diffusion = create_model_and_diffusion(**self.d)
        self.f_model.load_state_dict(th.load(f_model_path, map_location='cpu'))
        self.f_model.to(self.device)
        if use_fp16:  
            self.f_model.convert_to_fp16()
        self.f_model.eval()
        layer_name = get_last_conv_name(model)
        self.grad_cam = GradCamPlusPlus(model, layer_name)

    def cond_fn(self, x, t, y=None, mean=None, log_variance=None,  pred_xstart=None, 
            mask=0, **kwargs):
        time = int(t[0].detach().cpu())
        if mask is None: mask = 0
        if  self.range_t2_s <=time <= self.range_t2_e:
            eps = th.exp(0.5 * log_variance)
            delta = th.zeros_like(x)
            with th.enable_grad():
                delta.requires_grad_()
                for _ in range(self.nb_iter_conf):
                    tmpx = pred_xstart.detach() + delta  # range from -1~1
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

    def forward(self, inputs, labels, progress=False, adversarial=True):
        labels = th.tensor(labels).to(self.device)
        if inputs is None:
            sample = self.diffusion.p_sample_loop(
                lambda x, t, y, **kwargs: 
                    self.f_model(x, t, y),
                [len(labels), 3, 256, 256],
                clip_denoised=True,
                model_kwargs={"y":labels},
                cond_fn=self.cond_fn if adversarial else None,
                device=self.device,
                progress=progress,
            )
        else:
            assert not (inputs<0).any()
            img, labels = inputs.to(self.device), labels.to(self.device)
            mask = self.grad_cam(img).unsqueeze(1) if self.use_cam else 0
            sample = self.diffusion.p_sample_loop(
                lambda x, t, y, **kwargs: 
                    self.f_model(x, t, y),
                list(img.shape),
                clip_denoised=True,
                model_kwargs={"guide_x":img*2-1, 
                    "y":labels, "mask":mask,},
                cond_fn=self.cond_fn if adversarial else None,
                start_t=self.start_t,
                device=self.device,
                progress=progress, 
            )
        sample = th.clamp(sample,-1.,1.)
        return (sample+1)/2
    


class AdvFussionCIFAR10(AdvFussionAttack):
    def __init__(self, 
        model, 
        f_model_path, 
        adver_scale=0.01, 
        start_t=100, # 100
        range_t2_s=0, 
        range_t2_e=100,  # 100
        mask_p=1, 
        nb_iter_conf=30, 
        early_stop=True, 
        device='cuda', 
        use_cam=False, 
        **kwargs):
        super().__init__(model, adver_scale, start_t, 
            range_t2_s, range_t2_e, mask_p, nb_iter_conf,
            early_stop, device, use_cam)
        f_model = UNet(
            T=500, num_labels=10, ch=128, 
            ch_mult=[1, 2, 2, 2], num_res_blocks=2, 
            dropout=0.15).to(device)
        f_model.load_state_dict(th.load(f_model_path, map_location=device))
        f_model.eval()
        self.diffusion = GaussianDiffusionSampler(
            f_model, 1e-4, 0.028, 500, w=1.8).to(device)
        if use_cam:
            self.layer_name = get_last_conv_name(self.model)
    
    def get_config(self):
        return {
            'ts': self.range_t2_s, 
            'te': self.range_t2_e, 
            'nb_iter_conf': self.nb_iter_conf, 
            'adver_scale':self.adver_scale, 
        }

    def forward(self, inputs, labels, adversarial=True, contrastive=False):
        with th.no_grad(): 
            mask = 0
            model_kwargs = {"mask":mask, "modelConfig":self.get_config(),
                    "adversarial":adversarial, "contrastive":contrastive}
            if self.start_t == 500:
                assert isinstance(labels, th.Tensor)
                xt = th.randn(size=[len(labels), 3, 32, 32], device=self.device)
            else:
                assert not (inputs<0).any()
                img, labels = inputs.to(self.device), labels.to(self.device)
                if self.use_cam:
                    grad_cam = GradCamPlusPlus(self.model, self.layer_name)
                    mask = grad_cam(img).unsqueeze(1)
                xt = self.diffusion.get_xt(x_0=img*2-1, t=self.start_t-1)
            sample = self.diffusion(xt, labels, self.model, 
                    self.start_t, kwargs=model_kwargs)
            sample = th.clamp(sample,-1.,1.)
            return (sample+1)/2

if __name__ == '__main__':
    '''
    120w pic in imagenet traninig set, for imagenet-100, there's 12w pic
    90 epoch in adv_train, 100 batchsize(unimportance)
    ! Iterate through the training dataset once per epoch. 
    12w per epoch, 
    if we generate from scartch and keep the amount of data unchanged, which is xxxx 12w/50*411s 
    ---
    # batchsize could be 50 costing 32510MiB GPU(full), and cost 411.48651576042175s for generating AUES using reference images. 
    which meas only generate image should cost 11.41 days per epoch,
    ---
    Luckily, adv_train.py only train on adversarial example and wrong classified examples. 
    ---
    it'll take 1.5 times the GPU usage when using adversarial guidance. 
    '''
    from robustbench.utils import load_model
    from  torchvision import utils as vutils
    from time import time
    
    # model = load_model(model_name='Standard_R50', 
    #     dataset='imagenet', threat_model='Linf')
    # model = model.cuda().eval()
    # t1 = time()
    # model_path = '/root/256x256_diffusion.pt'
    # attacker =  AdvFussionImagNet(model, model_path)
    # labels = th.randint(0,999,[50])
    # x= th.randn(size=[len(labels), 3, 256, 256]).cuda()
    # x = th.clip(x+0.5, 0, 1)
    # sample = attacker(x, labels=labels, use_cam=False)
    # t2 = time()
    # th.save(sample, './test2.pt')
    # vutils.save_image(sample, './test2.jpg', normalize=True)
    # print("cost time: {}s".format((t2 - t1)))



    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    from torch.utils.data import DataLoader
    dataset = CIFAR10(
        root='/root/datasets', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=100, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True)
    model = load_model(model_name='Standard', 
        dataset='cifar10', threat_model='Linf')
    model = model.cuda().eval()
    model_path = '/root/DiffusionConditionWeight.pt'
    attacker =  AdvFussionCIFAR10(model, model_path)
    # bs = 1000 max gpu:30261MiB start_t=100 using 600s 
    t1 = time()
    natural_err_total = 0
    target_err_total = 0
    with th.no_grad(): 
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.cuda(), labels.cuda()
            pred_y = model(images).argmax(dim=1)
            natural_err_total += sum(labels!=pred_y)
            print(f"batch{i}:nature-{natural_err_total}")
            vutils.save_image(images, f'/root/trash/ori{i}.jpg')
            sample = attacker(images, labels)
            pred_y = model(sample).argmax(dim=1)
            target_err_total += sum(labels!=pred_y)
            print(f"batch{i}:adv-{target_err_total};")
            # th.save(sample, './test3.pt').
            vutils.save_image(sample, f'/root/trash/adv{i}.jpg')
            
    t2 = time()
    print("cost time: {} s".format((t2 - t1)))
