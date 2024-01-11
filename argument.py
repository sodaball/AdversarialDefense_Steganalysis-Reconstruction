import os
import argparse
import logging

def parser():

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--data_path',default='./datasets/cifar',type=str,help='path for input data')
    parser.add_argument('--num_epoches',default=10,type=int,help='number of total epochs')
    parser.add_argument('--batch_size',default=1,type=int,help='batch size') #默认16 修改后请修改attack中画图相关参数
    parser.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
    parser.add_argument('--ckpt_root',default='./ckpt',help='the directory to save the ckeckpoints')
    parser.add_argument('--epsilon',"-eps",default=0.05,type=float,help='the epsilon of FGSM')
    parser.add_argument('--advs_root',default='./samples/adv',help="the directory to save adversarial samples")
    parser.add_argument('--oris_root',default='./samples/ori',help="the directory to save original samples")
    parser.add_argument('--advf_root',default='./features/adv',help="the directory to save adversarial samples' features")
    parser.add_argument('--orif_root',default='./features/ori',help="the directory to save original samples' features")
    parser.add_argument('--image', '-i', default='clean',help="the image to classify")
    # parser.add_argument('--mode',default=train,type=str,help="mode of the program(train,attack,spam,fisher")

    return parser.parse_args()