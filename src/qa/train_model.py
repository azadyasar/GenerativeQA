import argparse
from qa.data import Vocabulary, Dataset
from qa.data.boolq_dataset import BoolQDataset
from qa.training import Trainer
from qa.training.configuration import TransformerModelConfig
from qa.util import get_device, get_dataset
import numpy as np

import logging
logger = logging.getLevelName("Trainer")

def train_qa_model(args: argparse.Namespace):
  vocab = Vocabulary()
  device = get_device()
  config = TransformerModelConfig(input_dim=len(vocab),
                                  output_dim=len(vocab),
                                  hid_dim=args.hid_dims,
                                  enc_layers=args.enc_layers,
                                  dec_layers=args.dec_layers,
                                  enc_heads=args.enc_heads,
                                  dec_heads=args.dec_heads,
                                  enc_pf_dim=args.enc_pf_dim,
                                  dec_pf_dim=args.dec_pf_dim,
                                  enc_dropout=args.enc_dropout,
                                  dec_dropout=args.dec_dropout,
                                  device=device,
                                  src_vocab=vocab,
                                  trg_vocab=vocab,
                                  max_passage_len=args.max_passage_len,
                                  max_question_len=args.max_question_len,
                                  max_answer_len=args.max_answer_len,
                                  batch_sz=args.batch_sz,
                                  save_model_path=args.save_model_path)
  trainer = Trainer(config=config,
                    learning_rate=args.lr,
                    weight_decay=args.wd_rate,
                    n_epochs=args.n_epochs,
                    clip=args.clip)
  
  train_dataset = get_dataset(args.train_dataset, **{
    'vocab': vocab,
    'device': device,
    'max_passage_len': config.max_passage_len,
    'max_question_len': config.max_question_len,
    'is_train': True
  })
  
  eval_dataset_name = args.eval_dataset if args.eval_dataset is not None and len(args.eval_dataset) > 0 else args.train_dataset
  eval_dataset = get_dataset(eval_dataset_name, **{
    'vocab': vocab,
    'device': device,
    'max_passage_len': config.max_passage_len,
    'max_question_len': config.max_question_len,
    'is_train': False
  })
  
  train_dataset.read_and_index()
  eval_dataset.read_and_index()

  trainer.train(train_dataset=train_dataset,
                eval_dataset=eval_dataset)
  logger.info("\n\nTraining completed. Evaluating model on the evaluation dataset.")
  test_loss = trainer.evaluate_(eval_dataset=eval_dataset)
  logger.info(f'\nTest Loss: {test_loss:.4f} |  Test PPL: {np.exp(test_loss):7.3f}')

def add_subparser(subparsers: argparse._SubParsersAction):
  parser = subparsers.add_parser('train', help='Train QA model')
  
  group = parser.add_argument_group('Dataset and vocabulary')
  group.add_argument('--train_dataset', required=True,
                     help='training dataset name')
  group.add_argument('--eval_dataset', required=False,
                     help='evaluation dataset name (if is not included in training dataset')
  group.add_argument('--test_dataset', required=False,
                     help='test dataset name')
  group.add_argument('--src_vocab', required=False,
                     help='source BPE model file path')
  group.add_argument('--trg_vocab', required=False,
                     help='target BPE model file path')
  
  group = parser.add_argument_group('Transformer model configurations')
  group.add_argument('--hid_dims', default=256, type=int,
                       help='hidden vector dimensions')
  group.add_argument('--enc_layers', default=4, type=int,
                       help='number of encoder layers')
  group.add_argument('--dec_layers', default=4, type=int,
                       help='number of decoder layers')
  group.add_argument('--enc_heads', default=8, type=int,
                       help='number of encoder attention heads')
  group.add_argument('--dec_heads', default=8, type=int,
                       help='number of decoder attention heads')
  group.add_argument('--enc_pf_dim', default=256*4, type=int,
                       help='encoder position-wise feed forward dimension. hid_dims * 4 is suggested.')
  group.add_argument('--dec_pf_dim', default=256*4, type=int,
                       help='decoder position-wise feed forward dimension. hid_dims * 4 is suggested.')
  group.add_argument('--max_passage_len', default=256, type=int,
                       help='maximum number of tokens allocated for the passage')
  group.add_argument('--max_question_len', default=32, type=int,
                       help='maximum number of tokens allocated for the question')
  group.add_argument('--max_answer_len', default=16, type=int,
                       help='maximum number of tokens allocated for the answer')
  
  group = parser.add_argument_group('Training specs')
  group.add_argument('--lr', default=0.0005, type=float,
                       help='learning rate')
  group.add_argument('--enc_dropout', default=0.25, type=float,
                       help='encoder dropout rate')
  group.add_argument('--dec_dropout', default=0.25, type=float,
                       help='decoder dropout rate')
  group.add_argument('--wd_rate', default=1e-4, type=float,
                       help='weight decay rate')
  group.add_argument('--clip', default=1., type=float,
                       help='gradient clipping')
  group.add_argument('--n_epochs', default=20, type=int,
                       help="number of epochs")
  group.add_argument('--batch_sz', default=128, type=int,
                       help="batch size")
  group.add_argument('--save_model_path', default='transformer_qa.pt',
                       help='save trained model to the file')
  
  parser.set_defaults(func=train_qa_model)