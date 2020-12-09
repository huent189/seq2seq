import torch
import argparse
import os
from model.transformers import Transformer
def train(config):
    vc = 200 # todo:
    os.environ['CUDA_VISIBLE_DEVICES']=config.gpu_id
    model = Transformer(vc)
    if config.gpu_id != -1:
        model = model.cuda()
    if config.pretrain_dir != "":
        model.load_state_dict(torch.load(config.pretrain_dir))
    optimizer = torch.optim.Adam(model.parameters(),)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshot")
    parser.add_argument('--pretrain_dir', type=str, default= "")
    parser.add_argument('--gpu_id', type=str, default='0')
    config = parser.parse_args()

