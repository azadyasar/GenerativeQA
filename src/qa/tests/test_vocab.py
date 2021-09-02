import unittest
from qa.data import Vocabulary

vocab = Vocabulary()

class TestConfig(unittest.TestCase):
  def test_should_create_vocab_object(self):

    self.assertIsNotNone(vocab)
    
  def test_should_return_positive_vocab_size(self):
    self.assertGreater(len(vocab), 0)
    
  def test_should_return_special_indices(self):
    self.assertIsNotNone(vocab.bos_idx)
    self.assertIsNotNone(vocab.eos_idx)
    self.assertIsNotNone(vocab.pad_idx)
    self.assertIsNotNone(vocab.cls_idx)
    self.assertIsNotNone(vocab.unk_idx)
    
  def test_should_encode_text(self):
    text = "this is a test"
    encoded = vocab.encode(text)
    self.assertIsNotNone(encoded)
    self.assertGreater(len(encoded), 0)
    
  def test_should_pad_encode_plus(self):
    text = "this is a short text"
    encoded_padded = vocab.encode_plus(text, 500)
    self.assertEqual(len(encoded_padded), 500)
    
  def test_should_correctly_decode_encoded_text(self):
    text = "this is a short text"
    encoded_padded = vocab.encode_plus(text, 500)
    decoded = vocab.decode(encoded_padded, skip_special_tokens=True)
    self.assertEqual(text, decoded)
    
  def test_should_convert_between_ids_tokens(self):
    token = '<CLS>'
    id = 32101
    
    converted_token = vocab.id_to_piece(id)
    converted_id = vocab.piece_to_id(token)
    self.assertEqual(token, converted_token)
    self.assertEqual(id, converted_id)
    
    
    
  # def test_should_load_model(self):
  #   model = torch.jit.load(config.transformer_path)
    
  #   self.assertIsNotNone(model)
    
  # def test_should_contain_bpe_path(self):
  #   bpe_path = config.bpe_path
    
  #   self.assertIsNotNone(bpe_path)
    
  # def test_should_bpe_model(self):
  #   bpe_model = sentencepiece.SentencePieceProcessor(config.bpe_path)
    
  #   self.assertIsNotNone(bpe_model)
    
  # def test_should_contain_bigram_path(self):
  #   bigram_path = config.bigram_path
    
  #   self.assertIsNotNone(bigram_path)
    
  # def test_should_load_bigram_model(self):
  #   bigram_model = open(config.bigram_path).readlines()
    
  #   self.assertIsNotNone(bigram_model) 
