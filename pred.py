import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import string
import pandas as pd
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
# from gensim.models import Word2Vec
from torch.nn.utils.rnn import pack_padded_sequence
from pathlib import Path
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,  Trainer, TrainingArguments, AdamW, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import GPTNeoForCausalLM, GPT2Tokenizer ,GPTNeoConfig
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel,BertTokenizer
from transformers import GPT2TokenizerFast
# from peft import LoraModel, LoraConfig
from pathlib import Path
import datetime
from tqdm import tqdm
import random
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import gc
import matplotlib.pyplot as plt

class Encoder(torch.nn.Module): #8,18,24 -> 8,40,24   (8x720 and 432x960)
    def __init__(self,h=128,n=8, e=64, a=4, o=1280):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(50257,e)
        # self.ip = nn.Sequential(
        #         nn.Linear(e,e//2),
        #         nn.ReLU(),
        #         nn.Linear(e//2,e)
        # )
        self.lstm = nn.LSTM(input_size=e,hidden_size=h,num_layers=n, batch_first=True, bidirectional=True)
        self.sa = nn.MultiheadAttention(h*2, a, dropout=0.1, batch_first=True)
        self.op = nn.Sequential(
                nn.Linear(2*h, h//2),
                nn.ReLU(),
                nn.Linear(h//2 , o),
        )
        # self.__init_weights()
        
    def forward(self, X): 
        emb = self.embed(X)  #bs,seq ,e
        # emb = self.ip(emb)
        enc, (hidden, cell) = self.lstm(emb) #bs, seq, h   #1,bs,h
        query = enc #nn.MA expects ; seq, bs, h
        atOp , atW = self.sa(query, query, query)
        #convert back to bs,seq, h
        # print(f'AtOp: {atOp.shape} | enc: {enc.shape}')
        logits = self.op(atOp + enc)
        # logits = self.op(enc)
        return logits , hidden , cell
        
    # def __init_weights(self):
    #     for module in [self.ip, self.op]:
    #         if isinstance(module, torch.nn.Linear):
    #             torch.nn.init.normal_(module.weight,mean = 0.0 , std=0.02)
    #             if module.bias is not None:
    #                 torch.nn.init.zeros_(module.bias)
    
    
class Decoder(torch.nn.Module):
    def __init__(self,h=128,n=8, e=64, a=4, o=50257):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(50257,e)
        # self.ip = nn.Sequential(
        #         nn.Linear(e,e),
        #         nn.ReLU(),
        #         nn.Linear(e,e)
        # )
        self.lstm = nn.LSTM(input_size=e,hidden_size=h,num_layers=n, batch_first=True, bidirectional=True)
        self.sa = nn.MultiheadAttention(h, a, dropout=0.1, batch_first=True)
        self.op = nn.Sequential(
                nn.Linear(2*h + e, h//2),
                nn.ReLU(),
                nn.Linear(h//2 , o),
        )
        # self.__init_weights()
        
    def forward(self, ip, ho, co, enc, mask): 
        emb = self.embed(ip)  #bs, seq_i, e
        # emb = self.ip(emb)
        dec, (ho, co) = self.lstm(emb, (ho, co)) #bs, seq_i, h   #1,bs,h
        query = emb #bs, seq_i, e 
        key = enc #bs, seq_e, o 
        value = enc #bs, seq_e, o 
        # print(f'Q:{query.shape} | K:{key.shape} | V:{value.shape}')
        atOp , atW = self.sa(query, key, value, key_padding_mask=mask) #bs, seq_i, e
        # print(f'Dec: {dec.shape} | atOp : {atOp.shape}')
        op = torch.cat([dec.squeeze(dim=1), atOp.squeeze(dim=1)], dim=1) #bs, seq_i, 2*h + bs, seq_i, e -> bs, 2*h + r 
        # op = torch.cat([ho[-1], co[-1], atOp.reshape(atOp.size(0), -1)], dim=-1)
        logits = self.op(op) #bs, o
        return logits, ho ,co 
        
    # def __init_weights(self):
    #     for module in [self.ip, self.op]:
    #         if isinstance(module, torch.nn.Linear):
    #             torch.nn.init.normal_(module.weight,mean = 0.0 , std=0.02)
    #             if module.bias is not None:
    #                 torch.nn.init.zeros_(module.bias)
    
    def init_state(self, batch_size):
        return (torch.zeros(2*self.n,batch_size, self.h).to(device),torch.zeros(2*self.n,batch_size, self.h).to(device))

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, seq_ip, ip_mask, seq_tg):
        enc, hidden, cell = self.encoder(seq_ip)
        outputs = []
        len_tg = seq_tg.shape[1]
        dec_ip = seq_tg[:,0].unsqueeze(dim=-1)
        # print('Target length: ')
        for t in range(1, len_tg):  # Teacher Forcing 
            op , hidden, cell = self.decoder(dec_ip, hidden, cell, enc, ip_mask)
            outputs.append(op)
            dec_ip = seq_tg[:,t].unsqueeze(dim=-1)
        torch.stack(outputs, dim=1)
        return outputs
    
def diverse_beam_search(decoder, encoder_output, ip_mask, hidden, cell, device, beam_width=5, diversity_penalty=0.7, max_len=100):
        dec_ip = torch.tensor([50256]).type(torch.int64).to(device)  # Start token
        beams = [(0.0, [dec_ip.item()], hidden.clone(), cell.clone())]  # (score, sequence, hidden, cell)
        count = 0
        for _ in range(max_len):
            all_candidates = []
            for score, seq, h, c in beams:
                if seq[-1] == 50256 and count > 0:  # EOS reached
                    all_candidates.append((score, seq, h, c))
                    continue
                dec_out, h_new, c_new = decoder(
                    torch.tensor([seq[-1]]).unsqueeze(0).to(device), h, c, encoder_output, ip_mask
                )
                log_probs = torch.nn.functional.log_softmax(dec_out, dim=-1)  # Shape: [1, vocab_size]
                top_k_log_probs, top_k_tokens = torch.topk(log_probs, beam_width, dim=-1)
                
                for i in range(beam_width):
                    new_score = score + top_k_log_probs[0, i].item() - (diversity_penalty * i)  # Diversity penalty
                    new_seq = seq + [top_k_tokens[0, i].item()]
                    all_candidates.append((new_score, new_seq, h_new.clone(), c_new.clone()))
                count = 1
            # Select top beam_width candidates
            beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            if all(seq[-1] == 50256 for _, seq, _, _ in beams):  # All beams ended
                break
        
        return beams[0][1]  # Return highest-scoring sequence

def mbr_decoding(decoder, encoder_output, ip_mask, hidden, cell, device, num_candidates=10, max_len=100):
        # Generate candidate sequences using top-k sampling
        candidates = []
        for _ in range(num_candidates):
            dec_ip = torch.tensor([50256]).type(torch.int64).to(device)
            seq = [dec_ip.item()]
            h, c = hidden.clone(), cell.clone()
            for _ in range(max_len):
                dec_out, h, c = decoder(dec_ip.unsqueeze(0), h, c, encoder_output, ip_mask)
                dec_ip = top_k_sampling(dec_out, k=5).unsqueeze(dim=0)  # Use top-k for diversity
                seq.append(dec_ip.item())
                if dec_ip.item() == 50256:
                    break
            candidates.append(seq)
        
        # Score candidates by similarity (e.g., average overlap with others)
        best_seq, best_score = None, float('-inf')
        for i, cand in enumerate(candidates):
            score = sum(sum(1 for t1, t2 in zip(cand, other) if t1 == t2) 
                        for other in candidates if other != cand) / (len(candidates) - 1)
            if score > best_score:
                best_score, best_seq = score, cand
        return best_seq

def top_k_sampling(logits, k=10, temperature=1.0):
        logits = logits / temperature  # Temperature scaling for diversity
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
        return top_k_indices[0, sampled_idx.item()]

def genOp(encoder, decoder, device, ip, ip_mask, mode='greedy', temperature=1.0, k=13, beam_width=5, diversity_penalty=0.7, num_candidates=10, max_len=100):
    encoder.eval()
    decoder.eval()
    # model.eval()
    print(f'\n\n\n GENOP FX CALL \n\n\n')
    with torch.no_grad():
        enc, hidden, cell = encoder(ip)
        print(f'Hidden : {hidden.shape} | Cell : {cell.shape}')
        if mode == 'greedy':
            outputs = []
            dec_ip = torch.tensor([50256]).type(torch.int64).to(device)
            count = 0
            while True:
                dec, hidden, cell = decoder(dec_ip.unsqueeze(dim=0), hidden, cell, enc, ip_mask)
                dec_ip = torch.argmax(dec, dim=-1)
                outputs.append(dec_ip.item())
                count += 1
                if count > max_len:
                    break
                if dec_ip.item() == 50256:
                    print('Self terminated !!!')
                    break
            return outputs
        elif mode=='sample':
            outputs = []
            dec_ip = torch.tensor([50256]).type(torch.int64).to(device)
            count = 0
            while True:
                dec, hidden, cell = decoder(dec_ip.unsqueeze(dim=0), hidden, cell, enc, ip_mask)
                # print(dec)
                dec = dec/temperature
                dec = torch.nn.functional.softmax(dec, dim=-1)
                dec_ip = torch.multinomial(input=dec, num_samples=1, replacement=True).squeeze(0)
                outputs.append(dec_ip.item())
                count += 1
                if count > max_len:
                    break
                if dec_ip.item() == 50256:
                    print('Self terminated !!!')
                    break
            return outputs
        elif mode=='top_k':
            outputs = []
            dec_ip = torch.tensor([50256]).type(torch.int64).to(device)
            count = 0
            while True:
                dec, hidden, cell = decoder(dec_ip.unsqueeze(dim=0), hidden, cell, enc, ip_mask)
                dec = torch.nn.functional.softmax(dec, dim=-1)
                top_k_probs , top_k_indices = torch.topk(dec, k, dim=-1)
                dec_ip = torch.multinomial(input=top_k_probs, num_samples=1, replacement=True).squeeze(0)
                dec_ip = top_k_indices[0, dec_ip.item()].unsqueeze(dim=0)
                outputs.append(dec_ip.item())
                count += 1
                if count > max_len:
                    break
                if dec_ip.item() == 50256:
                    print('Self terminated !!!')
                    break
            return outputs
        
        elif mode=='diverse-beam-search':
            outputs = diverse_beam_search(decoder, enc, ip_mask, hidden, cell, device, beam_width=beam_width, diversity_penalty=diversity_penalty)
            # print(f'GenOP stack trace: {outputs}')
            return outputs
        
        elif mode=='min-bayes-risk':
            outputs = mbr_decoding(decoder, enc, ip_mask, hidden, cell, device, num_candidates=num_candidates, max_len=max_len)
            return outputs

# ip = torch.tensor([[50256, 11195,   318, 13837,    11,  8272,   318,  2688,  4345,  1578,
#             11,  4475,   318,  3909,    11,  3035,   767,    11,  1941,   318,
#           4793,    11,  2435,   357,   315,    66,     8,   318,  1478,    25,
#            405,    11,  1078,   437,   590,   318,  3126,    11,  2931,    23,
#             11,  4080,   318, 24880, 10499,    11,  3576,    11,  4492,    11,
#          19316,   318,  4793,    12, 12726, 37985,  9952,  4041,    11,  6057,
#             62, 13376,   318, 19446,    11, 30408,   448,   318, 10352,    11,
#          11195,    62, 26675,   318,   657,    11,  8272,    62, 26675,   318,
#            352,    11, 11195,    62,    79, 49809,    47,   310,   318,  5598,
#           7441,  8272,    62,    79, 49809,    47,   310,   318,  4570,  7441,
#          11195,    62, 20910, 22093,   318,  1542,   357,  1314,   828,  8272,
#             62, 20910, 22093,   318,   718,   357,    20,   828, 11195,    62,
#             69, 42033,  6935,  2175,   318,   838,    13,    15,    11,  8272,
#             62,    69, 42033,  6935,  2175,   318,  1315,    13,    15,    11,
#          11195,    62, 36022,    34,  1371,   318,   657,    13,    15,    11,
#           8272,    62, 36022,    34,  1371,   318,   352,    13,    15,    11,
#          11195,    62,   445,    34,  1371,   318,   657,    13,    15,    11,
#           8272,    62,   445,    34,  1371,   318,   657,    13,    15,    11,
#          11195,    62,  8210,  1460,   318,   657,    13,    15,    11,  8272,
#             62,  8210,  1460,   318,   604,    13,    15,    11, 11195,    62,
#          26502, 41389,   364,   318,  1478,    13,    15,    11,  8272,    62,
#          26502, 41389,   364,   318,   352,    13,    15,    11, 11195,    62,
#             82,  3080,   318,   642,    13,    15,    11,  8272,    62,    82,
#           3080,   318,  1596,    13,    15,    11, 11195,    62,  1161,   318,
#          16185,    11,  8272,    62,  1161,   318, 16185,    11, 24623,   318,
#           3594,  9952,  4041,    11, 16060,    62, 15592,   318,   449,   641,
#          29921,  9038,    11, 17121,  7096,   292,    11,    42, 14057,  9852,
#           2634,    11, 10161, 18713, 12119,   280,  2634,    11, 35389, 26689,
#             75,  1012,   488,    88,    11, 30847, 11979,   406,    73,  2150,
#           3900,    11, 13787,   292, 10018, 17479,    11, 40747, 32371, 23720,
#             11, 15309, 38142,    81,   367,   293,    65,    11,    34,  3798,
#            376, 24247,    65,  2301,   292,    11, 10161, 18713,  1215,  1765,
#            323,   273,    11,  5124,  2731,   978,  6199,   544,    11, 49680,
#             68,   311,  2194,   418,    11,    41, 21356, 48590, 18226, 12523,
#             11,  4826,   280,  6031,  3930,    11, 31579, 44871, 12104,   324,
#          13235,    11,    32,  1014,    62, 15592,   318,  5199,  3469,    11,
#          22946,   292,  3169,   359,    11, 20191, 44677,    11, 13217,   261,
#          44312,    11, 14731, 14006,    11, 24338,  9740,  9860,    11, 25372,
#          20017,  9557,    11,    45, 47709,   797,    78,    12,    34, 11020,
#             11,  9704, 20833,    11,    33, 11369, 38343,  5799,    11, 26886,
#            418,  1665, 33425,    11, 32027, 21298,    11, 31306,  6559, 19574,
#           1040,    11, 30365, 13058,   273,    11, 25596,   271,  3248,    64,
#          10788,    68,    11,    42,   538,    64,    11,  7575,   318,  4153,
#              6]])
# ip_mask = torch.tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True]]) 

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# encoder = Encoder(h=64,n=2, e=64, a=4, o=64).to(device)
# decoder = Decoder(h=64,n=2, e=64, a=4, o=50257).to(device)
# model = Seq2Seq(encoder, decoder).to(device)

# # checkpoint = torch.load('./seq2seq_checkpoint.pt', weights_only=True, map_location=device)

# # model.load_state_dict(checkpoint['model_state_dict'])
# print(genOp(model.encoder, model.decoder, device, ip, ip_mask, mode='greedy', temperature=1.0, k=13, beam_width=5, diversity_penalty=0.7, num_candidates=10, max_len=100))