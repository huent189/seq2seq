import torch
import argparse
import os
from model.transformers import Transformer
from tqdm import tqdm
from dataloader import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data)


def train_one_epoch(model, data, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    epoch_bleu = 0
    for i, batch in tqdm(enumerate(data)):
        src = batch.en
        trg = batch.vi
        src = src.to(device)
        trg = trg.to(device)
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)
        trg_input = trg[:, :-1]
        optimizer.zero_grad()
        output = model(src, trg_input)

        # output = output[1:].view(-1, output.shape[-1])
        # trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_bleu += bleu_score(output, trg)
    return epoch_loss / len(data)

def evaluate(model, data, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data)):
            src = batch.en
            trg = batch.vi
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg)  # turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    if config.gpu_id != '-1':
        device = 'cuda'
    else:
        device = 'cpu'
    writer = SummaryWriter(log_dir=config.log_dir)
    train_data, val_data, en, vi = get_dataloader(config.data_dir, split=True, batch_size=config.batch_size, device=device)
    pad_idx = vi.vocab.stoi[vi.pad_token]
    model = Transformer(len(en.vocab.stoi))
    model = model.to(device)
    model.apply(initialize_weights)
    if config.pretrain_dir != "":
        model.load_state_dict(torch.load(config.pretrain_dir))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    # todo warm up cool down lr
    optimizer = torch.optim.Adam(model.parameters(), betas=[
                                 0.9, 0.98], lr=config.lr)
    for i in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_data, optimizer, criterion, config.grad_clip_norm, device)
        print(train_loss)
        writer.add_scalar('train', train_loss)
        val_loss = evaluate(model, val_data, criterion, device)
        writer.add_scalar('val loss', train_loss)
    # write log
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshot")
    parser.add_argument('--pretrain_dir', type=str, default="")
    parser.add_argument('--gpu_id', type=str, default='0')
    config = parser.parse_args()
    train(config)