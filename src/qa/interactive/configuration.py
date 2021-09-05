from qa.modeling import (Transformer,
                         Encoder,
                         Decoder)

from qa.util import count_parameters
from qa.data import Vocabulary
import torch

import logging
logger = logging.getLogger("TrainerConfig")

class TransformerModelConfig(object):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               hid_dim: int,
               enc_layers: int,
               dec_layers: int,
               enc_heads: int,
               dec_heads: int,
               enc_pf_dim: int,
               dec_pf_dim: int,
               enc_dropout: float,
               dec_dropout: float,
               device: str,
               src_vocab: Vocabulary,
               trg_vocab: Vocabulary,
               max_passage_len: int,
               max_question_len: int,
               max_answer_len: int,
               saved_model_path: str = 'transformer_qa.pt') -> None:
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hid_dim = hid_dim
    self.enc_layers = enc_layers
    self.dec_layers = dec_layers
    self.enc_heads = enc_heads
    self.dec_heads = dec_heads
    self.enc_pf_dim = enc_pf_dim
    self.dec_pf_dim = dec_pf_dim
    self.enc_dropout = enc_dropout
    self.dec_dropout = dec_dropout
    self.device = device
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.src_pad_idx = self.src_vocab.pad_idx
    self.trg_pad_idx = self.trg_vocab.pad_idx
    self.max_passage_len = max_passage_len
    self.max_question_len = max_question_len
    self.max_answer_len = max_answer_len
    self.saved_model_path = saved_model_path
    
    self.max_length = self.max_passage_len + self.max_question_len + 4
    
  def load_model(self):
    enc = Encoder(self.input_dim,
                  self.hid_dim,
                  self.enc_layers,
                  self.enc_heads,
                  self.enc_pf_dim,
                  self.enc_dropout,
                  self.device,
                  self.max_length)
    dec = Decoder(self.output_dim,
                  self.hid_dim,
                  self.dec_layers,
                  self.dec_heads,
                  self.dec_pf_dim,
                  self.dec_dropout,
                  self.device,
                  self.max_answer_len + 2)
    
    model = Transformer(encoder=enc,
                        decoder=dec,
                        src_pad_idx=self.src_pad_idx,
                        trg_pad_idx=self.trg_pad_idx,
                        device=self.device).to(self.device)
    
    logger.info(f"Constructed Transformer model with {count_parameters(model):,} trainable parameters. Loading model from {self.saved_model_path}")
    model.load_state_dict(torch.load(self.saved_model_path, map_location=self.device))
    logger.info(f"Loaded paremeters.")
    
    return model