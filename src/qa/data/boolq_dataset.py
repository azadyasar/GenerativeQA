from qa.data import Dataset, Vocabulary
import torch
from tqdm import tqdm
from datasets import load_dataset
from typing import List

import logging

from qa.data.dataset import Batch
logger = logging.getLogger("BoolQDataset")

BOOLQ_DATASET_NAME = 'boolq'

class BoolQDataset(Dataset):
  def __init__(self, vocab: Vocabulary, device: str,
               max_passage_len: int = 256,
               max_question_len: int = 32) -> None:
    super().__init__(vocab, device)
    self.dataset = load_dataset(BOOLQ_DATASET_NAME)
    self.train = self.dataset['train']
    self.validation = self.dataset['validation']
    self.max_passage_len = max_passage_len
    self.max_question_len = max_question_len
    
  def concat_passage_question(self, passage : str, question: str) -> List[int]:
    # len(tokenizer.encode_plus(boolq['train'][1]['passage'] + " okay " * 100 , padding='max_length', max_length=200, truncation=True)['input_ids'])
    result = [self.vocab.cls_idx] + self.vocab.encode_plus(passage, self.max_passage_len) + [self.vocab.sep_idx]
    result += self.vocab.encode_plus(question, self.max_question_len) + [self.vocab.sep_idx]
    return result
  
  def encode_answer(self, answer: str) -> List[int]:
    return [self.vocab.bos_idx] + self.vocab.encode(answer) + [self.vocab.eos_idx]
    
  def read_and_index(self):
    logger.info(f"Reading and indexing the dataset -{BOOLQ_DATASET_NAME}-")
    self.train_dataset_x = []
    self.train_dataset_y = []
    self.valid_dataset_x = []
    self.valid_dataset_y = []
    for instance in tqdm(self.train):
      p = instance['passage']
      q = instance['question']
      a = str(instance['answer'])
      encoded_pq = torch.tensor(self.concat_passage_question(p, q))
      encoded_a = torch.tensor(self.encode_answer(a))
      self.train_dataset_x.append(encoded_pq)
      self.train_dataset_y.append(encoded_a)
    logger.info(f"Indexed {len(self.train_dataset_x)} instances.")
      
  def __len__(self):
    if self.train_dataset_x:
      return len(self.train_dataset_x)
    return 0
    
  def generate(self, batch_sz: int) -> Batch:
    for i in range(0, len(self.train_dataset_x), batch_sz):
      src_tensor = self.pad_tensor(self.train_dataset_x[i:i+batch_sz], self.vocab.pad_idx)
      trg_tensor = self.pad_tensor(self.train_dataset_y[i:i+batch_sz], self.vocab.pad_idx)
      batch = Batch(src=src_tensor, trg=trg_tensor, device=self.device)
      
      yield batch
      