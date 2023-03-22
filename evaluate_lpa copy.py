
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
from perceptual_advex.perceptual_attacks import LagrangePerceptualAttack

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
    parser.add_argument('--batch_size', type=int, default=10,
                        help='number of examples/minibatch')
    # parser.add_argument('--parallel', type=int, default=2,
    #                     help='number of GPUs to train on')

    args = parser.parse_args()
    save_path = '/root/hhtpro/123/perceptual-advex/result_office'
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
    attack = LagrangePerceptualAttack(model, bound=0.5, lpips_model='alexnet')
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    for (img, label, img_name) in attack_loader:
        inputs = img.cuda()
        labels = label.cuda()

        adv_inputs = attack(inputs, labels)

        for k in range(args.batch_size):
            adv_inputs_cpu = adv_inputs[k, :, :, :].cpu()
            save_image(adv_inputs_cpu, image_path+'/'+ img_name[k][:-4]+'.jpg')
                