import unittest
from qa.data import Vocabulary, BoolQDataset, BoolQDataLoader
from qa.util.helpers import get_device

vocab = Vocabulary()
loader = BoolQDataLoader()
dataset = BoolQDataset(vocab=vocab,
                       data_loader=loader,
                       device=get_device(),
                       max_passage_len=64, max_question_len=16,
                       is_train=True)
dataset.read_and_index()

class TestConfig(unittest.TestCase):
  def test_should_create_vocab_object(self):

    self.assertIsNotNone(dataset)
    
  def test_should_read_index_dataset(self):
    # dataset.read_and_index()
    self.assertGreater(len(dataset), 0)
    
  def test_should_encode_passage_question_pair(self):
    p = "This case is meant to test whether the dataset can encode concat"
    q = "Can it though?"
    encoded = dataset.concat_passage_question(p, q)
    self.assertIsNotNone(encoded)
    self.assertGreater(len(encoded), 0)
    # cls + passage + sep + question + sep 
    self.assertEqual(len(encoded), 1 + dataset.max_passage_len + 1 + dataset.max_question_len + 1)
    self.assertEqual(encoded[0], vocab.cls_idx)
    self.assertEqual(encoded[dataset.max_passage_len + 1], vocab.sep_idx)
    self.assertEqual(encoded[-1], vocab.sep_idx)
    
  def test_should_generate_batch(self):
    batch = next(dataset.generate(16))
    
    self.assertIsNotNone(batch)
    self.assertEqual(len(batch.src), 16)
    self.assertEqual(len(batch.trg), 16)
    self.assertEqual(batch.src.device, get_device())