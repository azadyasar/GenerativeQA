from qa.data import Dataset, Vocabulary
import torch
from tqdm import tqdm
from datasets import load_dataset
from typing import List
import numpy
import logging

from qa.data import Batch
from qa.data.glue import BoolQDataLoader
logger = logging.getLogger("BoolQDataset")


class BoolQDataset(Dataset):
  def __init__(self, vocab: Vocabulary,
               data_loader: BoolQDataLoader,
               device: str,
               max_passage_len: int = 256,
               max_question_len: int = 32,
               is_train: bool = True) -> None:
    super().__init__(vocab, device)
    self.data_loader = data_loader
    self.is_train = is_train
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
    logger.info(f"Reading and indexing the dataset -{self.data_loader.name}- is_train: {self.is_train}")
    self.train_dataset_x = []
    self.train_dataset_y = []
    for instance in tqdm(self.data_loader.generate(self.is_train)):
      p = instance['passage']
      q = instance['question']
      a =  "yes it is true" if bool(instance['answer']) else "no it is not true"
      encoded_pq = torch.tensor(self.concat_passage_question(p, q))
      encoded_a = torch.tensor(self.encode_answer(a))
      self.train_dataset_x.append(encoded_pq)
      self.train_dataset_y.append(encoded_a)
    logger.info(f"Indexed {len(self.train_dataset_x)} instances.")
    self.train_dataset_x, self.train_dataset_y = BoolQDataset.shuffle(self.train_dataset_x, self.train_dataset_y)
    print("trainx[:2] = ", self.train_dataset_x[:2])
    print("trainy[:2] = ", self.train_dataset_y[:2])
      
  def shuffle(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]
    
  def __len__(self):
    if self.train_dataset_x:
      return len(self.train_dataset_x)
    return 0
    
  def generate(self, batch_sz: int) -> Batch:
    for i in range(0, len(self.train_dataset_x), batch_sz):
      src_tensor = self.pad_tensor(self.train_dataset_x[i:i+batch_sz], self.vocab.pad_idx)
      trg_tensor = self.pad_tensor(self.train_dataset_y[i:i+batch_sz], self.vocab.pad_idx)
      print('trg_tensor: ', trg_tensor)
      print('shape: ', trg_tensor.shape)
      batch = Batch(src=src_tensor, trg=trg_tensor, device=self.device)
      
      yield batch
      