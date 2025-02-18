from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(sequences: list[str], max_seq_len: int = 1000, size: str = "facebook/esm2_t30_150M_UR50D"): #max is 1000
    tokenizer = AutoTokenizer.from_pretrained(size)
    tokens = tokenizer(sequences, return_tensors="pt", truncation=True, padding=True, max_length=max_seq_len).to(device)
    return tokens

class esm2_model(nn.Module):
    def __init__(self, size: str = "facebook/esm2_t30_150M_UR50D"):
        super(esm2_model, self).__init__()
        self.esm = AutoModelForMaskedLM.from_pretrained(size)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids, attention_mask, output_hidden_states=True)
        return outputs
    
    def embed(self, input_ids, attention_mask):
        prot_features = self.esm(input_ids, attention_mask, output_hidden_states=True)['hidden_states'][-1]

        #remove cls and eos token from attention mask
        last_valid_indices = attention_mask.sum(dim=1) - 1
        for i in range(len(last_valid_indices)):
            attention_mask[i,0] = 0
            attention_mask[i,last_valid_indices[i]] = 0
        attention_mask = attention_mask.unsqueeze(-1).to(prot_features.dtype) #reshape to B x L x 1 

        prot_features = (prot_features * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        return prot_features

def init_model(size: str = "facebook/esm2_t30_150M_UR50D"):
    model = esm2_model(size=size).to(device)
    return model





