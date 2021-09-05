from qa.interactive import TransformerModelConfig
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List

import logging
logger = logging.getLogger("Generator")

import warnings
warnings.filterwarnings(action='ignore')

class Generator(object):
  def __init__(self,
               config: TransformerModelConfig):
    self.config = config
    self.model = config.load_model()
    
  def concat_passage_question(self, passage : str, question: str) -> List[int]:
    result = [self.config.src_vocab.cls_idx] + self.config.src_vocab.encode_plus(passage, self.config.max_passage_len) + [self.config.src_vocab.sep_idx]
    result += self.config.src_vocab.encode_plus(question, self.config.max_question_len) + [self.config.src_vocab.sep_idx]
    return result
  
  def encode_answer(self, answer: str) -> List[int]:
    return [self.config.trg_vocab.bos_idx] + self.config.trg_vocab.encode(answer) + [self.config.trg_vocab.eos_idx]
    
    
  def answer(self,
             passage: str,
             question: str,
             max_len: int = 100) -> str:
    self.model.eval()
    
    src_indexes = self.concat_passage_question(passage, question)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.config.device)
    src_mask = self.model.make_src_mask(src_tensor)
    
    with torch.no_grad():
      enc_src, self_attn = self.model.encoder.forward_w_attn(src_tensor, src_mask)
    
    trg_indices = [self.config.trg_vocab.bos_idx]
    
    for _ in range(max_len):
      trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(self.config.device)
      trg_mask = self.model.make_trg_mask(trg_tensor)
      with torch.no_grad():
        output, attention = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
      pred_token = output.argmax(2)[:, -1].item()
      print('pred_token: ', pred_token)
      trg_indices.append(pred_token)
      
      if pred_token == self.config.trg_vocab.eos_idx:
        break

    answer = self.config.trg_vocab.decode(trg_indices, skip_special_tokens=False)
    trg_tokens = [self.config.trg_vocab.id_to_piece(idx) for idx in trg_indices]
    return answer, trg_tokens[1:], attention, self_attn 
  
  def display_attention(self, passage, question, translation_tokens, attention, n_cols=4, figure_path = 'attention_figure.png'):
    n_heads = self.config.dec_heads
    n_rows = n_heads // n_cols
    
    if isinstance(passage, str):
      passage_question = [self.config.src_vocab.id_to_piece(idx) for idx in self.concat_passage_question(passage, question)]
    
    logger.info("Generating figure..")
    fig = plt.figure(figsize=(16, 16))
    plt.axis('off')
    
    for i in range(n_heads):
      ax = fig.add_subplot(n_rows, n_cols, i+1)
      
      _attention = attention.squeeze(0)[i].cpu().detach().numpy()
      cax = ax.matshow(_attention, cmap='bone')
      
      ax.tick_params(labelsize=12)
      ax.set_xticklabels([''] + [t for t in passage_question], rotation=45)
      ax.set_yticklabels([''] + translation_tokens)
      
      ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
      ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
      
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    logger.info(f"Saving figure to {figure_path}")
    plt.savefig(figure_path)