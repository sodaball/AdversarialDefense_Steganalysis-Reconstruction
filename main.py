import torch
import torch.nn as nn
import numpy as np
import os
from argument import parser

args = parser()

def gen_dir():
    os.makedirs(args.data_path,exist_ok=True)
    os.makedirs(args.ckpt_root,exist_ok=True)
    os.makedirs(os.path.join(args.advs_root,'eps={:}'.format(args.epsilon)),exist_ok=True)
    os.makedirs(args.oris_root,exist_ok=True)
    os.makedirs(os.path.join(args.advf_root,'eps={:}'.format(args.epsilon)),exist_ok=True)
    os.makedirs(args.orif_root,exist_ok=True)

if __name__ == '__main__':
    gen_dir()
    print('fin')