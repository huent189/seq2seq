import torchtext
from pyvi import ViTokenizer
import os
import dill
def tokenize(text):
    text = ViTokenizer.tokenize(text)
    return text.split()
def get_dataloader(root_path, split=False, batch_size=8, device='cuda', save_path=None, reload=None):
    text = torchtext.data.Field(sequential=True, tokenize=tokenize, lower=True)
    VI = torchtext.data.Field(tokenize=tokenize, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)
    EN = torchtext.data.Field(tokenize=tokenize, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)
    data_fields= [('en', EN), ('vi', VI)]
    if split:
        train_data,val_data = torchtext.data.TabularDataset.splits(path=root_path, train='train.csv', 
                        validation='val.csv', format='csv', fields=data_fields, skip_header=False)
        train_iter, val_iter = torchtext.data.BucketIterator.splits(
                            (train_data, val_data), # we pass in the datasets we want the iterator to draw data from
                            batch_sizes=(batch_size, batch_size),
                            device=device, # if you want to use the GPU, specify the GPU number here
                            sort_key=lambda x: len(x.en), # the BucketIterator needs to be told what function it should use to group the data.
                            sort_within_batch=False,
                            shuffle=True # we pass repeat=False because we want to wrap this Iterator layer.
                            )
    else:
        train_data = torchtext.data.TabularDataset(path=root_path, format='csv', fields=data_fields, skip_header=True)
        train_iter= torchtext.data.BucketIterator(
                            (train_data, val_data), # we pass in the datasets we want the iterator to draw data from
                            batch_size=batch_size,
                            device=device, # if you want to use the GPU, specify the GPU number here
                            sort_key=lambda x: len(x.en), # the BucketIterator needs to be told what function it should use to group the data.
                            sort_within_batch=False,
                            shuffle=True # we pass repeat=False because we want to wrap this Iterator layer.
                            )
    if reload:
        with open(os.path.join(reload,"VI.Field"),"rb")as f:
            VI = dill.load(f)
        with open(os.path.join(reload,"EN.Field"),"rb")as f:
            EN = dill.load(f)
    else:
        VI.build_vocab(train_data)
        EN.build_vocab(train_data)
    if save_path:
        with open(os.path.join(save_path,"VI.Field"),"wb")as f:
            dill.dump(VI, f)
        with open(os.path.join(save_path,"EN.Field"),"wb")as f:
            dill.dump(EN, f)
    if split:
        return train_iter, val_iter, EN, VI 
    else:
        return train_iter, EN, VI  
if __name__ == "__main__":
    train, val, en, vi = get_dataloader('/home/dell/Documents/thesis/transformer/data', split=True, reload='/home/dell/Documents/thesis/transformer/snapshot')
    for i, batch in enumerate(train):
        print(batch)
        break

        