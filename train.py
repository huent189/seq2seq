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


def train_one_iter(model, data, optimizer, criterion, clip, device, trg_init_idx):
    # model.train()
    # epoch_loss = 0
    # global count
    # for i, batch in tqdm(enumerate(data)):
    src = data.en
    trg = data.vi
    src = src.to(device)
    trg = trg.to(device)
    # trg_input = torch.zeros_like(trg).to(device)
            # trg_input[:, 1:] = trg[:, 0:-1]
            # trg_input[:, 0] = trg_init_idx
    optimizer.zero_grad()
    output = model(src, trg[:,:-1])  # turn off teacher forcing
    output = output.reshape(-1, output.shape[-1])
    trg = torch.reshape(trg[:, 1:], [-1])
    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    # epoch_loss += loss.item()
    # count = count + 1
    # if count % 100 == 0:
    #     writer.add_scalar('train_per_iter', loss.item(), count // 100)
    #     translate_sentence(src[0], en, vi, model, device)

    # return epoch_loss / len(data)
    return loss.item()


def evaluate(model, data, criterion, device, trg_init_idx):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data)):
            src = batch.en
            trg = batch.vi
            src = src.to(device)
            trg = trg.to(device)
            # trg_input = torch.zeros_like(trg).to(device)
            # trg_input[:, 1:] = trg[:, 0:-1]
            # trg_input[:, 0] = trg_init_idx
            output = model(src, trg)  # turn off teacher forcing
            output = output.reshape(-1, output.shape[-1])
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    if config.gpu_id != '-1':
        device = 'cuda'
    else:
        device = 'cpu'
    train_data, val_data, en, vi = get_dataloader(
        config.data_dir, batch_size=config.batch_size, device=device)
    src_pad_idx = en.vocab.stoi[en.pad_token]
    trg_pad_idx = vi.vocab.stoi[vi.pad_token]
    print('vocab size: en:', len(en.vocab.stoi), 'vi:', len(vi.vocab.stoi))
    model = Transformer(max(len(en.vocab.stoi), len(
        vi.vocab.stoi)), src_pad_idx, trg_pad_idx)
    model = model.to(device)
    model.apply(initialize_weights)
    print('Model parameter: ', count_parameters(model))
    if config.pretrain_model != "":
        model.load_state_dict(torch.load(config.pretrain_model))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    # todo warm up cool down lr
    optimizer = NoamOpt(512, 0.1, 2000, torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    best_loss = 100
    count = 0
    for i in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        for i, batch in tqdm(enumerate(train_data)):
            train_loss = train_one_iter(
                model, batch, optimizer, criterion, config.grad_clip_norm, device, vi.vocab.stoi[vi.init_token])
            epoch_loss += train_loss
            count += 1
            if count % config.snapshot_iter == 0:
                torch.save(model.state_dict(), os.path.join(
                    config.snapshots_folder, "Epoch_" + str(i) + '.pth'))
                val_loss = evaluate(model, val_data, criterion, device, vi.vocab.stoi[vi.init_token])
                writer.add_scalar('val loss', val_loss, i)
                if val_loss < best_loss:
                    torch.save(model.state_dict(), os.path.join(
                        config.snapshots_folder, "best.pth"))
            if count % (config.snapshot_iter // 10) == 0:
                writer.add_scalar('train', epoch_loss /
                                  (config.snapshot_iter // 10), count)
                epoch_loss = 0
                translate_sentence(batch.en[0], en, vi, model, device)
                model.train()


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
    parser.add_argument('--gpu_id', type=str, default='0')
    config = parser.parse_args()
    writer = SummaryWriter(log_dir=config.log_dir)
    seed_all(2345)
    train(config)
