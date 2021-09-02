from os import truncate
import sentencepiece as spm
from typing import List
from transformers import T5Tokenizer

class Vocabulary(object):
  def __init__(self):
    self.tokenizer = T5Tokenizer.from_pretrained('t5-base') #spm.SentencePieceProcessor(tokenizer_path)
    self.tokenizer.add_special_tokens({
      'bos_token': "<s>",
      'cls_token': "<CLS>",
      'sep_token': "<SEP>"
    })
    
  def encode(self, sentence: str) -> List[int]:
    return self.tokenizer.encode(sentence)
  
  def encode_plus(self, sentence: str, max_length: int) -> List[int]:
    return self.tokenizer.encode_plus(sentence, padding='max_length', max_length=max_length, truncation=True)['input_ids']
    
  def encode_and_pack(self, sentence: str) -> List[str]:
    result =  [self.bos_idx] + self.encode(sentence) + [self.eos_idx]
    
    return result
  
  def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
    return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
  
  def decode_with_check(self, tokens: List[int]) -> str:
    return self.tokenizer.DecodeIdsWithCheck(tokens)
  
  def id_to_piece(self, id: int) -> str:
    return self.tokenizer.convert_ids_to_tokens(id)
  
  def piece_to_id(self, piece: str) -> int:
    return self.tokenizer.convert_tokens_to_ids(piece)
  
  @property
  def bos_idx(self) -> int:
    return self.tokenizer.bos_token_id
  
  @property
  def eos_idx(self) -> int:
    return self.tokenizer.eos_token_id
  
  @property
  def unk_idx(self) -> int:
    return self.tokenizer.unk_token_id
  
  @property
  def pad_idx(self) -> int:
    return self.tokenizer.pad_token_id
  
  @property
  def sep_idx(self) -> int:
    return self.tokenizer.sep_token_id
  
  @property
  def cls_idx(self) -> int:
    return self.tokenizer.cls_token_id
  
  def __len__(self) -> int:
    return self.tokenizer.vocab_size
  