import torchtext
from pyvi import ViTokenizer
import os
# import dill
import torch
from torchtext.data.utils import get_tokenizer
import html
import string
# table = str.maketrans('', '', string.punctuation)
def vi_tokenize(text):
    text = html.unescape(text)
    # text = text.translate(table)
    text = ViTokenizer.tokenize(text)
    text = text.split()
    # print(text)
    return text

def en_tokenize(text):
    # text = html.unescape(text)
    # return get_tokenizer('spacy')(text)
    # text = text.translate(table)
    text = text.split()
    # print(text)
    return text

def get_dataloader(root_path, train_file="train_200k_r.csv", test_file="val_200k_r.csv", batch_size=8, device='cuda', save_path=None, reload=None):
    VI = torchtext.data.Field(tokenize=vi_tokenize,
                              init_token='<sos>',
                              eos_token='<eos>',
                              lower=True,
                              fix_length=51,
                              batch_first=True)
    EN = torchtext.data.Field(tokenize=en_tokenize,
                              init_token='<sos>',
                              eos_token='<eos>',
                              lower=True,
                              fix_length=50,
                              batch_first=True)
    data_fields = [('vi_no_accents', EN), ('vi', VI)]
    train_data, val_data = torchtext.data.TabularDataset.splits(path=root_path, train=train_file, validation=test_file, format='csv', fields=data_fields, skip_header=True)
    train_iter, val_iter = torchtext.data.BucketIterator.splits(
        # we pass in the datasets we want the iterator to draw data from
        (train_data, val_data),
        batch_sizes=(batch_size, batch_size),
        device=device,  # if you want to use the GPU, specify the GPU number here
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_key=lambda x: len(x.vi_no_accents),
        sort_within_batch=False,
        # we pass repeat=False because we want to wrap this Iterator layer.
        shuffle=True
    )
    if reload is not None:
        VI = torch.load(os.path.join(reload, "VI.Field"))
        EN = torch.load(os.path.join(reload, "EN.Field"))
    else:
        print('build vocab')
        VI.build_vocab(train_data, min_freq=2)
        EN.build_vocab(train_data, min_freq=2)
    if save_path:
        torch.save(VI, os.path.join(save_path, "VI.Field"))
        torch.save(EN, os.path.join(save_path, "EN.Field"))
    return train_iter, val_iter, EN, VI


if __name__ == "__main__":
    str = "It &apos;s a big community . It &apos;s such a big community"
    unescaped = html.unescape(str)
    print(unescaped)
    print(get_tokenizer('spacy')(unescaped))
