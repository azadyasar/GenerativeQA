import unittest
from qa.data import BoolQDataLoader

loader = BoolQDataLoader()

class TestConfig(unittest.TestCase):
  def test_should_create_loader_object(self):

    self.assertIsNotNone(loader)
    
  def test_should_generate_instance_for_train(self):
    # dataset.read_and_index()
    instance = next(loader.generate(is_train=True))
    self.assertIsNotNone(instance)
    self.assertTrue(instance['passage'])
    self.assertTrue(instance['question'])
    self.assertTrue(instance['answer'])
    
  def test_should_generate_instance_for_validation(self):
    # dataset.read_and_index()
    instance = next(loader.generate(is_train=False))
    self.assertIsNotNone(instance)
    self.assertTrue(instance['passage'] is not None)
    self.assertTrue(instance['question'] is not None)
    self.assertTrue(instance['answer'] is not None)