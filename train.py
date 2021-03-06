import torch
import argparse
import os
from model.transformers import Transformer
from tqdm import tqdm
from dataloader import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from optimizer import NoamOpt
# from torchtext.data.metrics import bleu_score
from util import seed_all
from test import translate_sentence


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data)

count = 0
def train_one_iter(model, train_data, optimizer, criterion, clip, device, en, vi):
    global count
    model.train()
    epoch_loss = 0
    for i, data in tqdm(enumerate(train_data)):
      src = data.vi_no_accents
      trg = data.vi
      # optimizer.zero_grad()
      src = src.to(device)
      trg = trg.to(device)
    
      output = model(src, trg[:,:-1])  # turn off teacher forcing
      output = output.contiguous().view(-1, output.shape[-1])
      trg = trg[:,1:].contiguous().view(-1)
      loss = criterion(output, trg)
      # if ((i + 1) % 8 == 0):
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
      optimizer.step()
      optimizer.zero_grad()
      epoch_loss += loss.item()
      count += 1
      if count % 400 == 0:
        writer.add_scalar('train per iter', loss.item(), count)
        print('train_per_iter:', loss.item())
        model.translate_sentence(data.vi_no_accents[0], en, vi, 50)
        model.train()
    return epoch_loss /len(train_data)


def evaluate(model, data, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data):
            src = batch.vi_no_accents
            trg = batch.vi
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:,:-1])  # turn off teacher forcing
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config):
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    if config.gpu_id != '-1':
        device = 'cuda'
    else:
        device = 'cpu'
    if config.load_vocab is not None:
        train_data, val_data, en, vi = get_dataloader(
            config.data_dir, batch_size=config.batch_size, device=device, reload=config.load_vocab)
    else:
        train_data, val_data, en, vi = get_dataloader(
            config.data_dir, batch_size=config.batch_size, device=device, save_path=config.snapshots_folder)
    # train_data, val_data, en, vi = get_dataloader(config.data_dir, batch_size=config.batch_size, device=device)
    src_pad_idx = en.vocab.stoi[en.pad_token]
    trg_pad_idx = vi.vocab.stoi[vi.pad_token]
    print('vocab size: en:', len(en.vocab.stoi), 'vi:', len(vi.vocab.stoi))
    model = Transformer(len(en.vocab.stoi), len(
        vi.vocab.stoi), src_pad_idx, trg_pad_idx, device, d_model=256, n_layers=5)
    model = model.to(device)
    model.apply(initialize_weights)
    print('Model parameter: ', count_parameters(model))
    if config.pretrain_model != "":
        model.load_state_dict(torch.load(config.pretrain_model))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    # todo warm up cool down lr
    optimizer = NoamOpt(512, 1, 2000, torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    best_loss = 100
    # count = 0
    # epoch_loss = 0
    for i in range(config.num_epochs):
        # model.train()
        # for j, batch in tqdm(enumerate(train_data)):
        train_loss = train_one_iter(
                model, train_data, optimizer, criterion, config.grad_clip_norm, device, en, vi)
        writer.add_scalar('train', train_loss, i)
        print('train_avg:', train_loss)
        
            # epoch_loss += train_loss
            # count += 1
        # if count % config.snapshot_iter == 0:
        torch.save(model.state_dict(), os.path.join(
            config.snapshots_folder, "Epoch_" + str(i) + '.pth'))
        val_loss = evaluate(model, val_data, criterion, device)
        writer.add_scalar('val loss', val_loss, i)
        print('val_loss:', val_loss)
        if val_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(
                config.snapshots_folder, "best.pth"))


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
    parser.add_argument('--display_iter', type=int, default=40)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshot")
    parser.add_argument('--pretrain_model', type=str, default="")
    parser.add_argument('--load_vocab', type=str)
    parser.add_argument('--gpu_id', type=str, default='0')
    config = parser.parse_args()
    writer = SummaryWriter(log_dir=config.log_dir)
    seed_all(18921)
    train(config)
