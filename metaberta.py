from KmerTokenizer import KmerTokenizer
from transformers import AutoModel
import torch

###
#pip install git+https://github.com/MsAlEhR/KmerTokenizer.git
###


def tokenizer(sequences, max_seq_len):
    tokenizr = KmerTokenizer(kmerlen=6, overlapping=True, maxlen=max_seq_len)
    tokenized_sequences = []
    for seq in sequences:
        tokens = tokenizer.kmer_tokenize(seq)
        tokenized_sequences.append(tokens) 
    return tokenized_sequences


def init_model():
    model = AutoModel.from_pretrained("MsAlEhR/MetaBERTa-bigbird-gene")
    return model
