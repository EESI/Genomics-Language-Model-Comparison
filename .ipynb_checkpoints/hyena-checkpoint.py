from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(sequences, max_seq_len=160_000): #max is 160_000
    tokenizer = AutoTokenizer.from_pretrained('LongSafari/hyenadna-medium-160k-seqlen-hf', trust_remote_code=True)
    tokenized_sequences = []
    for seq in sequences:
        tokens = tokenizer(seq, return_tensors="pt", padding="max_length", max_length=max_seq_len)["input_ids"].to(device)
        tokenized_sequences.append(tokens)
    return tokenized_sequences

def init_model():
    model = AutoModelForSequenceClassification.from_pretrained(
            "LongSafari/hyenadna-medium-160k-seqlen-hf", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    return model
