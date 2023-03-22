
from typing import Dict, List
import torch
import csv
import argparse

from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model
from perceptual_advex.attacks import *
from robustbench import load_model
from robustness.datasets import DATASETS

'''
attacks = [
        NoAttack(),
        LinfAttack(model, dataset_name='cifar', num_iterations=100),
        L2Attack(model, dataset_name='cifar', num_iterations=100),
        JPEGLinfAttack(model, dataset_name='cifar', num_iterations=100),
        FogAttack(model, dataset_name='cifar', num_iterations=100),
        StAdvAttack(model, num_iterations=100),
        ReColorAdvAttack(model, num_iterations=100),
        LagrangePerceptualAttack(model, num_iterations=200),
        PerceptualPGDAttack(model, num_iterations=200, lpips_model='alexnet')
    ]
'''

exp = lambda i:f"/root/hhtpro/123/perceptual-advex/data/exp/exp{i}/exp{i}.ckpt.pth"
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adversarial training evaluation')
    parser.add_argument('--expnum', type=int, required=True, help='checkpoint path')
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='model architecture')
    parser.add_argument('--dataset', type=str, default='cifar',
                        help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='/root/hhtpro/123/CIFAR10',
                        help='path to datasets directory')
    parser.add_argument('--attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--parallel', type=int, default=1,
                        help='number of GPUs to train on')
    parser.add_argument('--num_batches', default=5, type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--per_example', action='store_true', default=False,
                        help='output per-example accuracy')
    parser.add_argument('--output', default='evaluation.csv', type=str, help='output CSV')
    
    args = parser.parse_args()
    args.output = f'/root/hhtpro/123/perceptual-advex/data/exp/exp{args.expnum}/evaluation{args.num_batches}.csv'

    if args.expnum in [1, 2, 3, 4]:
        args.checkpoint = exp(args.expnum)
        print(args.checkpoint)
        dataset, model = get_dataset_model(args)
        _, val_loader = dataset.make_loaders(1, args.batch_size, only_val=True)
    else:
        args.checkpoint = exp(1)
        dataset, model = get_dataset_model(args)
        _, val_loader = dataset.make_loaders(1, args.batch_size, only_val=True)
        if args.expnum == 5:
            print("load Linf Rebuffi2021Fixing_70_16_cutmix_extra")
            model = load_model(model_name="Rebuffi2021Fixing_70_16_cutmix_extra", 
            dataset='cifar10', threat_model="Linf", model_dir='/root/hhtpro/123/models')
        elif args.expnum == 6:
            print("load L2 Rebuffi2021Fixing_70_16_cutmix_extra")
            model = load_model(model_name="Rebuffi2021Fixing_70_16_cutmix_extra", 
            dataset='cifar10', threat_model="L2", model_dir='/root/hhtpro/123/models')
        else:
            raise RuntimeError()
            
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    # args.attacks = [
    #     "NoAttack()",
    #     "LinfAttack(model, dataset_name='cifar', num_iterations=100)",
    #     "L2Attack(model, dataset_name='cifar', num_iterations=100)",
    #     "JPEGLinfAttack(model, dataset_name='cifar', num_iterations=100)",
    #     "FogAttack(model, dataset_name='cifar', num_iterations=100)",
    #     "StAdvAttack(model, num_iterations=100)",
    #     "ReColorAdvAttack(model, num_iterations=100)",
    #     "LagrangePerceptualAttack(model, num_iterations=40, lpips_model='alexnet')",
    #     # "PerceptualPGDAttack(model, num_iterations=40, lpips_model='alexnet')"
    # ]
    args.attacks = [
        
    ]
    attack_names: List[str] = args.attacks
    attacks = [eval(attack_name) for attack_name in attack_names]

    # Parallelize
    if torch.cuda.is_available():
        device_ids = list(range(args.parallel))
        model = nn.DataParallel(model, device_ids)
        attacks = [nn.DataParallel(attack, device_ids) for attack in attacks]

    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}

    for batch_index, (inputs, labels) in enumerate(val_loader):
        print(f'BATCH {batch_index:05d}')

        if (
            args.num_batches is not None and
            batch_index >= args.num_batches
        ):
            break

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        for attack_name, attack in zip(attack_names, attacks):
            adv_inputs = attack(inputs, labels)
            with torch.no_grad():
                adv_logits = model(adv_inputs)
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}',
                  sep='\t')
            batches_correct[attack_name].append(batch_correct)

    print('OVERALL')
    accuracies = []
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              sep='\t')
        accuracies.append(accuracy)

    with open(args.output, 'w') as out_file:
        out_csv = csv.writer(out_file)
        out_csv.writerow(attack_names)
        if args.per_example:
            for example_correct in zip(*[
                attacks_correct[attack_name] for attack_name in attack_names
            ]):
                out_csv.writerow(
                    [int(attack_correct.item()) for attack_correct
                     in example_correct])
        out_csv.writerow(accuracies)
