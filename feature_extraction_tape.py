import re
import numpy as np
import torch
from tqdm.auto import tqdm
from tape.tokenizers import TAPETokenizer

model = torch.load('/share/home/grp-lizy/pangyx/Experiments/KPTDP/pretrained_models/tape/tape-bert-base.pkl').to('cuda')
tokenizer = TAPETokenizer(vocab='iupac')

def read_protein_sequences(file):
    with open(file) as f:
        sequences = f.read()
        
    sequences = list(filter(None, sequences.split('\n'))) 
    sequences = [re.sub(r"[\*]", "X", seq) for seq in sequences]
    return(sequences)


def bert_encodings(sequences):
    outputs = []
    for seq in tqdm(sequences):
        encoded_input = tokenizer.encode(seq)
        encoded_input =torch.tensor([encoded_input]).to('cuda')
        pooled_output = model(encoded_input)[1].detach()
        outputs.append(pooled_output)
    outputs = torch.cat(outputs, axis=0).cpu().numpy()
    return outputs


def write_to_npy(encodings, file):
    np.save(file, encodings)
    
    
if __name__ == "__main__":

    file = "./data/DeepAFP-main-train.txt"
    sequences=read_protein_sequences(file)
    encode=bert_encodings(sequences)
    write_to_npy(encode, "DeepAFP-main-train.npy")