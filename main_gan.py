import os
import argparse
from solver_encoder_gan import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config.batch_size, config.len_crop, config.sv2ttsemb)
    
    solver = Solver(vcc_loader, config)

    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=0.1, help='weight for hidden code loss')#0.1
    parser.add_argument('--dim_neck', type=int, default=32)#32
    parser.add_argument('--dim_emb', type=int, default=512)#256#
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    
    # Training configuration.
    #parser.add_argument('--data_dir', type=str, default='./spmel')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=3000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    parser.add_argument('--model_id', type=str, default='autovc_500spk_ssim_spkmean_adamw_160', help='model name')
    parser.add_argument('--sv2ttsemb', action="store_true", help='sv2tts emb or autovc emb')
    parser.add_argument('--autoVC_path', type=str, default='/mnt/hdd0/hsiaohan/result/checkpoint/autovc_20spk_autovcmel_l1_16k_repar/G.ckpt', help='pretrain model path')
    parser.add_argument('--no_repar', action="store_true", help='have repararmeterization or not')
    parser.add_argument('--no_attn', action="store_true", help='have attention or not')
    parser.add_argument('--no_dis', action="store_true", help='have discriminator or not')
    parser.add_argument('--mel', action="store_true", help='input mel or not')

    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)