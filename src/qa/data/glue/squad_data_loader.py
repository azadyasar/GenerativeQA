from datasets import load_dataset

import logging

logger = logging.getLogger('SquadDataLoader')

SQUAD_DATASET_NAME = 'squad'

class SquadDataLoader(object):
  def __init__(self) -> None:
    super().__init__()
    self.dataset = load_dataset(SQUAD_DATASET_NAME)
    self.train = self.dataset['train']
    self.validation = self.dataset['validation']
    
  def generate(self, is_train: bool = True) -> object:
    target_dataset = self.train if is_train else self.validation
    for instance in target_dataset:
      yield instance
      
  @property
  def name(self) -> str:
    return SQUAD_DATASET_NAME
    