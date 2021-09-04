from abc import abstractmethod
import torch
from typing import List
from qa.data.vocabulary import Vocabulary

class Batch(object):
  def __init__(self, src: torch.Tensor, trg: torch.Tensor, device):
    self.src = src.to(device)
    self.trg = trg.to(device)
    
class Dataset(object):
  
  def __init__(self,
               vocab: Vocabulary,
               device: str) -> None:
      super().__init__()
      self.vocab = vocab
      self.device = device
  
  @abstractmethod
  def __len__(self):
    pass
  
  @abstractmethod
  def generate(self, batch_sz: int) -> Batch:
    pass
  
  def pad_tensor(self,
                 sequences: List[torch.Tensor],
                 pad_idx: int) -> torch.Tensor:
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(pad_idx)
    for i, tensor in enumerate(sequences):
      length = tensor.size(0)
      out_tensor[i, :length] = tensor
    return out_tensor