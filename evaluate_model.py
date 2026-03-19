import torch
import argparse
import logging
import test
import os
import open_clip
import datasets1
from model_try2 import Encoder
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default = 'fashion200k', help = "dataset type")
parser.add_argument('--fashioniq_split', default = 'val-split')
parser.add_argument('--fashion200k_path', default = '.')
parser.add_argument('--fashioniq_path', default = ".")
parser.add_argument('--shoes_path', default = ".")
parser.add_argument('--cirr_path', default = ".")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--hidden_dim', type=int, default=512)

parser.add_argument('--P', type=int, default=3)
parser.add_argument('--tau_', type=float, default=0.1)
parser.add_argument('--lambda_', type=float, default=1.0) 
parser.add_argument('--eta_', type=float, default=1.0) 
parser.add_argument('--kappa_', type=float, default=0.5)
parser.add_argument('--wc', type=int, default=2) 
parser.add_argument('--N_p', type=int, default=2) 
parser.add_argument('--weighted', type=bool, default=True)
parser.add_argument('--ckpt_path', default = "./")
args = parser.parse_args()

#{P,wc,N_P}: CIRR:{1,2,1} Shoes&FashionIQ:{3,2,2} Fashion200K:{1,2,2}


def load_dataset():
    clip_path = '.'#"/root/CLIP-ViT-B-32-laion2B-s34B-b79K" #
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained=os.path.join(clip_path, 'open_clip_pytorch_model.bin'))
    if args.dataset == 'fashioniq':
        print('Reading fashioniq')
        fashioniq_dataset = datasets1.FashionIQ(path = args.fashioniq_path, transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        return [fashioniq_dataset]
    elif args.dataset == 'shoes':
        print('Reading shoes')
        shoes_dataset = datasets1.Shoes(path = args.shoes_path, transform = [preprocess_train, preprocess_val])
        return [shoes_dataset]
    elif args.dataset == 'cirr':
        print('Reading cirr')
        cirr_dataset = datasets1.CIRR(path = args.cirr_path, transform = [preprocess_train, preprocess_val])
        return [cirr_dataset]
    elif args.dataset == 'fashion200k':
        print('Reading fashion200k')
        fashion200k_dataset = datasets1.Fashion200k(path = args.fashion200k_path, split = 'train', transform = [preprocess_train, preprocess_val])
        fashion200k_testset = datasets1.Fashion200k(path = args.fashion200k_path, split = 'test', transform = [preprocess_train, preprocess_val])
        return [fashion200k_dataset, fashion200k_testset]


model = torch.load(args.ckpt_path)

Encoder_model = Encoder(hidden_dim=args.hidden_dim, dropout=args.dropout_rate, local_token_num=args.P, t = args.tau_, wc = args.wc, N_p = args.N_p, weighted=args.weighted)
Encoder_model.load_state_dict(model, strict=False)
Encoder_model = Encoder_model.cuda()

dataset_list = load_dataset()

if args.dataset in ['shoes']:
    with torch.no_grad():
        t = test.test(args, Encoder_model, dataset_list[0], args.dataset)
        print(t)
if args.dataset in ['fashioniq']:
    for ci, category in enumerate(['dress', 'shirt', 'toptee']):
        t = test.test(args, Encoder_model, dataset_list[0], category)
        print(t)
elif args.dataset in ['fashion200k']:
    fashion200k_testset = dataset_list.pop(-1)
    t = test.test_fashion200k_dataset(args, Encoder_model, fashion200k_testset)
    print(t)
elif args.dataset in ['cirr']:
    t = test.test_cirr_valset(args, Encoder_model, dataset_list[0])
    print(t)