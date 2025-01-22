import torch
from transformers import BertModel, BertConfig
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras
import keras_bert
from itertools import product
import numpy as np
from random import randint


def keras2torch(kmodel,
                params={'embed_dim': 250, 'seq_len': 502, 'transformer_num': 12,
                        'head_num': 5, 'feed_forward_dim': 1024,
                        'dropout_rate': 0.05, 'vocab_size': 69}):
    tmodel = BertModel(BertConfig(vocab_size=params['vocab_size'],
                                  hidden_size=params['embed_dim'],
                                  num_attention_heads=params['head_num'],
                                  num_hidden_layers=params['transformer_num'],
                                  intermediate_size=params['feed_forward_dim'],
                                  hidden_dropout_prob=params['dropout_rate'],
                                  attention_probs_dropout_prob=params['dropout_rate'],
                                  max_position_embeddings=params['seq_len'],
                                  layer_norm_eps=tf.keras.backend.epsilon() * tf.keras.backend.epsilon()))
    # set torch model tensors to the ones from the keras model
    td = {t[0]: t[1] for t in tmodel.named_parameters()}
    kd = {t.name: t for t in kmodel.weights}
    def set_tensor(tname, karray):
        assert (tshape:=td[tname].detach().numpy().shape) == (
            kshape:=karray.shape), f'{tname} has incompatible shape: {tshape} != {kshape}'
        with torch.no_grad():
            td[tname].data = torch.nn.Parameter(torch.Tensor(karray))
    # 1 INPUT
    t_pfix = 'embeddings.'
    k_pfix = 'Embedding-'
    # set_tensor(t_pfix + 'position_ids', td[t_pfix + 'position_ids']) # don't change
    set_tensor(t_pfix + 'word_embeddings.weight', kd[k_pfix + 'Token/embeddings:0'].numpy())
    set_tensor(t_pfix + 'position_embeddings.weight', kd[k_pfix + 'Position/embeddings:0'].numpy())
    set_tensor(t_pfix + 'token_type_embeddings.weight', kd[k_pfix + 'Segment/embeddings:0'].numpy())
    set_tensor(t_pfix + 'LayerNorm.weight', kd[k_pfix + 'Norm/gamma:0'].numpy())
    set_tensor(t_pfix + 'LayerNorm.bias', kd[k_pfix + 'Norm/beta:0'].numpy())
    # 2 LAYERS
    for i in range(params['transformer_num']):
        t_pfix_l = f'encoder.layer.{i}.'
        k_pfix_l = f'Encoder-{i+1}-'
        # SELF-ATTENTION
        # NOTE: (embed_dim x embed_dim) matrices have to be transposed!
        t_pfix = t_pfix_l + 'attention.'
        k_pfix = k_pfix_l + f'MultiHeadSelfAttention/Encoder-{i+1}-MultiHeadSelfAttention_'
        set_tensor(t_pfix + 'self.query.weight', kd[k_pfix + 'Wq:0'].numpy().transpose())
        set_tensor(t_pfix + 'self.query.bias', kd[k_pfix + 'bq:0'].numpy())
        set_tensor(t_pfix + 'self.key.weight', kd[k_pfix + 'Wk:0'].numpy().transpose())
        set_tensor(t_pfix + 'self.key.bias', kd[k_pfix + 'bk:0'].numpy())
        set_tensor(t_pfix + 'self.value.weight', kd[k_pfix + 'Wv:0'].numpy().transpose())
        set_tensor(t_pfix + 'self.value.bias', kd[k_pfix + 'bv:0'].numpy())
        set_tensor(t_pfix + 'output.dense.weight', kd[k_pfix + 'Wo:0'].numpy().transpose())
        set_tensor(t_pfix + 'output.dense.bias', kd[k_pfix + 'bo:0'].numpy())
        # NORM
        t_pfix = t_pfix_l + 'attention.output.LayerNorm.'
        k_pfix = k_pfix_l + f'MultiHeadSelfAttention-Norm/'
        set_tensor(t_pfix + 'weight', kd[k_pfix + 'gamma:0'].numpy())
        set_tensor(t_pfix + 'bias', kd[k_pfix + 'beta:0'].numpy())
        # FF
        t_pfix = t_pfix_l + ''
        k_pfix = k_pfix_l + 'FeedForward'
        set_tensor(t_pfix + 'intermediate.dense.weight',
                   kd[k_pfix + f'/Encoder-{i+1}-FeedForward_W1:0'].numpy().transpose())
        set_tensor(t_pfix + 'intermediate.dense.bias', kd[k_pfix + f'/Encoder-{i+1}-FeedForward_b1:0'].numpy())
        set_tensor(t_pfix + 'output.dense.weight',
                   kd[k_pfix + f'/Encoder-{i+1}-FeedForward_W2:0'].numpy().transpose())
        set_tensor(t_pfix + 'output.dense.bias', kd[k_pfix + f'/Encoder-{i+1}-FeedForward_b2:0'].numpy())
        set_tensor(t_pfix + 'output.LayerNorm.weight', kd[k_pfix + '-Norm/gamma:0'].numpy())
        set_tensor(t_pfix + 'output.LayerNorm.bias', kd[k_pfix + '-Norm/beta:0'].numpy())
    # 3 OUTPUT (before class)
    set_tensor('pooler.dense.weight', kd['NSP-Dense/kernel:0'].numpy().transpose())
    set_tensor('pooler.dense.bias', kd['NSP-Dense/bias:0'].numpy())
    return tmodel


def load_bert(bert_path, compile_=False):
    """get bert model from path"""
    custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                      'GlorotUniform': keras.initializers.glorot_uniform}
    custom_objects.update(keras_bert.get_custom_objects())
    model = keras.models.load_model(bert_path, compile=compile_,
                                    custom_objects=custom_objects)
    return model




def get_token_dict(alph='ACGT', k=3) -> dict:
    """get token dictionary dict generated from `alph` and `k`"""
    token_dict = keras_bert.get_base_dict()
    for word in [''.join(_) for _ in product(alph, repeat=k)]:
        token_dict[word] = len(token_dict)
    return token_dict


def seq2kmers(seq, k=3, stride=3, pad=True, to_upper=True):
    """transforms sequence to k-mer sequence.
    If specified, end will be padded so no character is lost"""
    if (len(seq) < k):
        return [seq.ljust(k, 'N')] if pad else []
    kmers = []
    for i in range(0, len(seq) - k + 1, stride):
        kmer = seq[i:i + k]
        if to_upper:
            kmers.append(kmer.upper())
        else:
            kmers.append(kmer)
    if (pad and len(seq) - (i + k)) % k != 0:
        kmers.append(seq[i + k:].ljust(k, 'N'))
    return kmers


def seq2tokens(seq, token_dict, seq_length=250, max_length=None,
               k=3, stride=3, window=True, seq_len_like=None):
    """transforms raw sequence into list of tokens to be used for
    fine-tuning BERT"""
    if (max_length is None):
        max_length = seq_length
    if (seq_len_like is not None):
        seq_length = min(max_length, np.random.choice(seq_len_like))
        # open('seq_lens.txt', 'a').write(str(seq_length) + ', ')
    seq = seq2kmers(seq, k=k, stride=stride, pad=True)
    if (window):
        start = randint(0, max(len(seq) - seq_length - 1, 0))
        end = start + seq_length - 1
    else:
        start = 0
        end = seq_length
    indices = [token_dict['[CLS]']] + [token_dict[word]
                                       if word in token_dict
                                       else token_dict['[UNK]']
                                       for word in seq[start:end]]
    if (len(indices) < max_length):
        indices += [token_dict['']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    segments = [0 for _ in range(max_length)]
    return [np.array(indices), np.array(segments)]



def tokenizer(sequences, max_seq_len):
    tokend= get_token_dict()
    tokenized_sequences = []
    for seq in sequences:
        tokens, _ = seq2tokens(seq, tokend, int(np.ceil(max_seq_len / 3 + 1)))  # Tokenize each sequence
        tokenized_sequences.append(tokens)  # Collect only the tokens
    return tokenized_sequences


  
def init_model():
    # Load the Keras model from .h5
    keras_model = load_bert('bertax.h5')
    # Define the parameters for the BERT model
    params = {
        'embed_dim': 250,
        'seq_len': 502,
        'transformer_num': 12,
        'head_num': 5,
        'feed_forward_dim': 1024,
        'dropout_rate': 0.05,
        'vocab_size': 69
    }
    
    # Convert the Keras model to PyTorch
    model = keras2torch(keras_model, params)

    return model

