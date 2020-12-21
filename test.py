import torch
from dataloader import vi_tokenize, en_tokenize, get_dataloader
import argparse
import os
from model.transformers import Transformer
from tqdm import tqdm
from util import seed_all

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    if isinstance(sentence, str):
        toks = en_tokenize(sentence.lower())
        toks = [src_field.init_token] + toks + [src_field.eos_token]
        token_idecies = [src_field.vocab.stoi[tok] for tok in toks]
        src_tensor = torch.LongTensor(token_idecies).unsqueeze(0).to(device)
    else: 
        print('input', [src_field.vocab.itos[s] for s in sentence])
        src_tensor = sentence.unsqueeze(0)
    trg = model.translate_sentence(src, src_field, trg_field, 50)

def accuracy_score(pred, trg):
    pred = np.array(pred)
    trg = np.array(trg)
    return (pred == trg).sum() / pred.shape[0]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--pretrain_model', type=str, default="")
    parser.add_argument('--manual_train', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    config = parser.parse_args()
    seed_all(18921)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    if config.manual_train:
        print('----------MANUAL MODE--------------')
    else:
        print('----------BATCH MODE--------------')
    if config.gpu_id != '-1':
        device = 'cuda'
    else:
        device = 'cpu'
    print('Building vocab....')
    train_data, test_data, en, vi = get_dataloader(
        config.data_dir, batch_size=config.batch_size, device=device, test_file="test_250k.csv")
    src_pad_idx = en.vocab.stoi[en.pad_token]
    trg_pad_idx = vi.vocab.stoi[vi.pad_token]
    print('en', len(en.vocab.stoi), 'vi', len(vi.vocab.stoi))
    print('Vocab has been built')
    print('Loading pretrain model...')
    model = Transformer(max(len(en.vocab.stoi), len(
        vi.vocab.stoi)), src_pad_idx, trg_pad_idx)
    model = model.to(device)
    model.load_state_dict(torch.load(config.pretrain_model))
    model.eval()
    print('Model has been loaded')
    if config.manual_train:
        while(True):
            print("Enter an english sentence:")
            en_sentence = input()
            pred = translate_sentence(en_sentence, en, vi, model, device)
            print('predict output:', "".join(pred))
    else:
        with open(os.path.join(config.log_dir, 'output.txt'), 'w') as f:
            num_sample = 0
            acc = 0
            for i, batch in tqdm(enumerate(train_data)):
                src = batch.en
                trg = batch.vi
                src = src.to(device)
                trg = trg.to(device)
                # print(src.shape)
                for en_sentence in src:
                    print(en_sentence)
                    pred = translate_sentence(en_sentence, en, vi, model, device)
                    print('predict output:', "".join(pred))
                    f.write("".join(pred) + "\n")
                    acc += accuracy_score(pred, trg)
                    num_sample += 1
                    break
                print('Accuracy: ', acc / num_sample)
                # output = model(src, trg)
                # print()
                # pred_token = output[0].argmax(1)
                # print(pred_token)
                # print(trg[0])
                break