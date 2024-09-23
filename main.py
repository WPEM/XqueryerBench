import torch
import random
import argparse
import torch.nn.functional as F
import numpy as np
import os
from train import train
import datetime
import wandb
import json



import wandb

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--model', type=str, default="Mamba",
                        help='Model name')
    parser.add_argument('--seed', type=int, default=100, metavar='N',
                        help='the rand seed')
    parser.add_argument('--weight', action="store_true")
    parser.add_argument('--focal', action="store_true")
    parser.add_argument('--label_smoothing', action="store_true")
    parser.add_argument('--datapath', type=str, default="/data/zzn/scp")
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Num workers in dataloader')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='learning rate')
    parser.add_argument('--patience', type=int, default=3,
                        help='patience for early stopping')
    parser.add_argument('--warmup-epochs', default=2, type=int, metavar='N',
                        help='number of warmup epochs')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    #### only For time-series models ####
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=128,
                        help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=64,
                        help='dimension of fcn')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropou')
    parser.add_argument('--factor', type=int, default=5,
                        help='attn factor')
    parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
    parser.add_argument('--patch_len', type=int, default=64, help='patch length')
    parser.add_argument('--stride', type=int, default=64, help='patch stride')
    parser.add_argument('--activation', type=str, default='gelu', 
                        help='activation')
    parser.add_argument('--output_attention', action='store_true', default=False, 
                        help='whether to output attention in ecoder')
    parser.add_argument('--llm_layers', type=int, default=3, 
                        help='num of the LLm layers')
    parser.add_argument('--mlp', type=int, default=0, 
                        help='set 1 to tune the MLP in GPT4TS')
    parser.add_argument('--seq_len', type=int, default=3500, 
                        help='The length of XRD')
    parser.add_argument('--pred_len', type=int, default=3500, help='prediction sequence length')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--seg_len', type=int, default=50,
                        help='the length of segmen-wise iteration of SegRNN')


    args = parser.parse_args()
    



    if args.seed < 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed
    set_seed(seed)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.login()
    wandb.init(
        project="XRDBench_prl_rruff",
        config=args.__dict__,
        name=nowtime
        )
    output=train(args,nowtime)
    wandb.finish()
    