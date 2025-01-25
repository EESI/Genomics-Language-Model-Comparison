import torch
from transformers import AutoTokenizer, AutoModel, BertConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(sequences, max_seq_len=512): #max is 512
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    tokenized_sequences = []
    for seq in sequences:
        tokens = tokenizer(seq, return_tensors="pt", padding="max_length", max_length=max_seq_len, truncation=True).to(device)
        tokenized_sequences.append(tokens) 
    return tokenized_sequences

def init_model():
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", config=config, trust_remote_code=True).to(device)
    return model
