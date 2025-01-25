import torch
from transformers import AutoModel, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(sequences, max_seq_len=1000): #max is 1000
    tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True)
    tokenized_sequences = []
    for seq in sequences:
        tokens = tokenizer([seq], return_tensors='pt')
        tokenized_sequences.append(tokens)
    return tokenized_sequences

def init_model():    
    model = AutoModel.from_pretrained('tattabio/gLM2_650M', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    return model
