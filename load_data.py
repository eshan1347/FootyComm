import pandas as pd
import torch
# import numpy as np
# import torch
from transformers import GPT2Tokenizer

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# train_size = 50000 
# test_size = 2500
# val_size = 2500

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token

# df = pd.read_csv('./prepro_data.csv')

# train_df = df[:train_size]
# test_df = df[train_size:train_size + test_size + val_size]

# test_df.to_csv('test_data.csv')
# print('Test df saved...')

test_df = pd.read_csv('./test_data.csv')
test_df = test_df.reset_index(drop=True)
# print(test_df.index)

class TextDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
    
        def __len__(self):
            return len(self.X)
    
        # def __getitem__(self, idx):
        #     return self.X[idx]['input_ids'], self.X[idx]['attention_mask'] , self.y[idx]['input_ids'], self.y[idx]['attention_mask']
    
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
def collate_fn(batch):
    X = [i[0] for i in batch] 
    y = [i[1] for i in batch] 

    lenX = []
    maxlen = max([len(tokenizer.tokenize(i)) for i in X])
    maylen = max([len(tokenizer.tokenize(i)) for i in y])

    # print(f'maxlen: {maxlen} | maylen: {maylen}')

    inputs = [tokenizer(i, max_length=maxlen, padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True) for i in X]
    targets = [tokenizer(i, max_length=maylen, padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True) for i in y]

    input_ids, input_mask = [], []
    for i in inputs:
        input_ids.append(i['input_ids'])
        input_mask.append(i['attention_mask'])
    target_ids, target_mask = [], []
    for i in targets:
        target_ids.append(i['input_ids'])
        target_mask.append(i['attention_mask'])
    
    return (torch.vstack(input_ids), torch.vstack(input_mask), torch.vstack(target_ids), torch.vstack(target_mask))

val_ds = TextDataset(test_df['X'].values, test_df['y'].values)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=5000, shuffle=False, collate_fn=collate_fn)

# print(test_df.head())

def get_sample(i, device='cpu'):
    # X,y = test_df['X'][idx], test_df['y'][idx]
    # tok_X = tokenizer(X, return_tensors='pt', return_attention_mask=True)
    # tok_y = tokenizer(y, return_tensors='pt', return_attention_mask=True)
    # return X,y, tok_X, tok_y
    # return X,y
    val_batch = next(iter(valloader))
    return val_batch[0][i].unsqueeze(dim=0).to(device), val_batch[1][i].unsqueeze(dim=0).type(torch.float32).to(device), val_batch[2][i].to(device), val_batch[3][i].to(device)


# X, y, tok_X, tok_y = get_sample(1)
# print(f'X: {X} \n y: {y}')
# print(type(tok_X))
# print(tok_X)
# print(tok_X.shape, tok_y.shape)
