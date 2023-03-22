
from typing import Dict, List
import torch
import csv
import argparse
from torch import nn
from torch.nn import functional as F
from pytorch_msssim import ssim
import lpips
from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model
from perceptual_advex.perceptual_attacks import get_lpips_model, LPIPSDistance, MarginLoss, PROJECTIONS, normalize_flatten_features
class LagrangePerceptualAttack(nn.Module):
    def __init__(self, model, bound=0.5, step=None, num_iterations=20,
                 binary_steps=5, h=0.1, kappa=1, lpips_model='self',
                 projection='newtons', decay_step_size=True,
                 num_classes=None,
                 include_image_as_activation=False,
                 randomize=False, random_targets=False):

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        self.decay_step_size = decay_step_size
        self.num_iterations = num_iterations
        if step is None:
            if self.decay_step_size:
                self.step = self.bound
            else:
                self.step = self.bound * 2 / self.num_iterations
        else:
            self.step = step
        self.binary_steps = binary_steps
        self.h = h
        self.random_targets = random_targets
        self.num_classes = num_classes

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(
            self.lpips_model,
            include_image_as_activation=include_image_as_activation,
        )
        self.loss = MarginLoss(kappa=kappa, targeted=self.random_targets)
        self.projection = PROJECTIONS[projection](self.bound, self.lpips_model)

    def threat_model_contains(self, inputs, adv_inputs):
        """
        Returns a boolean tensor which indicates if each of the given
        adversarial examples given is within this attack's threat model for
        the given natural input.
        """

        return self.lpips_distance(inputs, adv_inputs) <= self.bound

    def _attack(self, inputs, labels):
        perturbations = torch.zeros_like(inputs)
        perturbations.normal_(0, 0.01)
        perturbations.requires_grad = True

        batch_size = inputs.shape[0]
        step_size = self.step

        lam = 0.01 * torch.ones(batch_size, device=inputs.device)

        input_features = normalize_flatten_features(
            self.lpips_model.features(inputs)).detach()

        live = torch.ones(batch_size, device=inputs.device, dtype=torch.bool)

        for binary_iter in range(self.binary_steps):
            if live.sum() != 0:
                for attack_iter in range(self.num_iterations):
                    if self.decay_step_size:
                        step_size = self.step * \
                            (0.1 ** (attack_iter / self.num_iterations))
                    else:
                        step_size = self.step

                    if perturbations.grad is not None:
                        perturbations.grad.data.zero_()

                    adv_inputs = (inputs + perturbations)[live]

                    if self.model == self.lpips_model:
                        adv_features, adv_logits = \
                            self.model.features_logits(adv_inputs)
                    else:
                        adv_features = self.lpips_model.features(adv_inputs)
                        adv_logits = self.model(adv_inputs)

                    adv_labels = adv_logits.argmax(1)
                    adv_loss = self.loss(adv_logits, labels[live])
                    adv_features = normalize_flatten_features(adv_features)
                    lpips_dists = (adv_features - input_features[live]).norm(dim=1)
                    all_lpips_dists = torch.zeros(batch_size, device=inputs.device)
                    all_lpips_dists[live] = lpips_dists

                    loss = -adv_loss + lam[live] * F.relu(lpips_dists - self.bound)
                    loss.sum().backward()

                    grad = perturbations.grad.data[live]
                    grad_normed = grad / \
                        (grad.reshape(grad.size()[0], -1).norm(dim=1)
                        [:, None, None, None] + 1e-8)

                    dist_grads = (
                        adv_features -
                        normalize_flatten_features(self.lpips_model.features(
                            adv_inputs - grad_normed * self.h))
                    ).norm(dim=1) / self.h

                    updates = -grad_normed * (
                        step_size / (dist_grads + 1e-8)
                    )[:, None, None, None]

                    perturbations.data[live] = (
                        (inputs[live] + perturbations[live] +
                        updates).clamp(0, 1) -
                        inputs[live]
                    ).detach()

                    if self.random_targets:
                        live[live.clone()] = (adv_labels != labels[live]) | (lpips_dists > self.bound)
                    else:
                        live[live.clone()] = (adv_labels == labels[live]) | (lpips_dists > self.bound)
                    if live.sum() == 0:
                        break
            adv_inputs = (inputs + perturbations).detach()
            adv_inputs = self.projection(inputs, adv_inputs, input_features)
            yield adv_inputs
            lam[all_lpips_dists >= self.bound] *= 10

    def forward(self, inputs, labels):
        return self._attack(inputs, labels)

import pandas as pd
import numpy as np
from PIL import Image 
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from robustbench.utils import load_model
from torchvision.utils import save_image
import os
import perceptual_advex.logger as logger

class MyCustomDataset(Dataset):
    def __init__(self, inputs_path = "images"):
        # Preprocess
        self.to_tensor = transforms.ToTensor()
        self.image_name = np.asarray(os.listdir(inputs_path))
        self.data_len = self.image_name.shape[0]
        self.inputs_path = inputs_path

    def __getitem__(self, index):
        single_image_name = self.image_name[index]
        inputs_as_inputs = Image.open(os.path.join(self.inputs_path, single_image_name)) 

        inputs_as_tensor = self.to_tensor(inputs_as_inputs)
        
        single_image_label = int(single_image_name.split('.')[0])

        return (inputs_as_tensor, single_image_label, single_image_name)

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adversarial training evaluation')

    add_dataset_model_arguments(parser, include_checkpoint=True)
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    # parser.add_argument('--parallel', type=int, default=2,
    #                     help='number of GPUs to train on')

    args = parser.parse_args()
    save_path = '/root/hhtpro/123/perceptual-advex/result_modify'
    image_path = save_path + '/img'
    logger.configure(save_path)
    loader = MyCustomDataset(inputs_path="/root/hhtpro/123/GA-Attack-main/data/images")
    sampler = torch.utils.data.SequentialSampler(loader)
    attack_loader = torch.utils.data.DataLoader(dataset=loader,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                sampler=sampler, num_workers=8, pin_memory=True)
    model = load_model(model_name='Engstrom2019Robustness', dataset='imagenet', threat_model='Linf')
    model = model.to('cuda').eval()
    # attack = LagrangePerceptualAttack(model, bound=0.5, num_iterations=1,
    #             binary_steps=13, lpips_model='alexnet', h=0.001)
    attack = LagrangePerceptualAttack(model, bound=0.5, lpips_model='alexnet')
    binary_list = [int(i+1) for i in range(attack.binary_steps)]
    for bin in binary_list:
        p = image_path+ "/" + f"{str(bin)}"
        if not os.path.exists(p):
            os.mkdir(p)
    # Parallelize
    # if torch.cuda.is_available():
    #     device_ids = list(range(args.parallel))
    #     model = nn.DataParallel(model, device_ids)
    #     attack = nn.DataParallel(attack, device_ids)

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.cuda()
    
    save_dict = {str(i):{'acc':0, 'Lp':0, 'LPIPS':0, 'SSIM':0} for i in binary_list}
    count = 0
    for (img, label, img_name) in attack_loader:
        inputs = img.cuda()
        labels = label.cuda()

        attack_g = attack(inputs, labels)
        for i in binary_list:
            adv_inputs = next(attack_g)
            with torch.no_grad():
                adv_logits = model(adv_inputs)
                err_mask = (adv_logits.argmax(1) != labels).detach()
                lpips_batch = loss_fn_alex.forward(inputs, adv_inputs, normalize=True)[err_mask]
                acc = err_mask.float().sum().item()
                Lp_dis = torch.abs(adv_inputs - inputs).reshape(len(adv_inputs), -1).max(dim = -1)[0][err_mask]
                lpips_batch = loss_fn_alex.forward(inputs, adv_inputs, normalize=True)[err_mask]
                ssim_batch = ssim(inputs, adv_inputs, data_range=1., size_average=False)[err_mask]

            # torch.cuda.synchronize()
            # save infomation at i. such as success rate/ Lp dis/ LPIPS/ SSIM/   adv_inputs
            save_dict[str(i)]['acc'] += acc
            save_dict[str(i)]['Lp'] += Lp_dis.data.sum().detach().clone()
            save_dict[str(i)]['LPIPS'] += lpips_batch.sum().detach().clone()
            save_dict[str(i)]['SSIM'] += ssim_batch.sum().detach().clone()

        for k in range(args.batch_size):
            adv_inputs_cpu = adv_inputs[k, :, :, :].cpu()
            save_image(adv_inputs_cpu, image_path+ "/" + f"{str(i)}" + '/'+ img_name[k][:-4]+'.jpg')
                

        for k, eps in enumerate(binary_list):
            logger.log('attack {}'.format(count))
            logger.log('eps at: {}'.format(eps))
            logger.log('target_err_total at: {}'.format(save_dict[str(eps)]['acc']))
            logger.log('Avg Lp_distance: {}'.format((save_dict[str(eps)]['Lp']/ save_dict[str(eps)]['acc'])))
            logger.log('Avg LPIPS dis: {}'.format((save_dict[str(eps)]['LPIPS']/ save_dict[str(eps)]['acc'])))
            logger.log('Avg SSIM dis: {}'.format((save_dict[str(eps)]['SSIM']/ save_dict[str(eps)]['acc'])))
            logger.log('=========')
        count += 1

    torch.save(save_dict, save_path+ "/" + "save_dict.pt")

