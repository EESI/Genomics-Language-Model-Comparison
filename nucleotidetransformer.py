from transformers import AutoTokenizer, AutoModel
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(sequences, max_seq_len=1000): #max is 1000
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")    
    tokenized_sequences = []
    for seq in sequences:
        tokens = tokenizer(seq, return_tensors="pt", padding="max_length", max_length=max_seq_len, truncation=True).to(device)
        tokenized_sequences.append(tokens)
    return tokenized_sequences

def init_model():    
    model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species").to(device)
    return model
