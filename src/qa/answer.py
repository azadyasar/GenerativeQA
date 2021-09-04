import argparse
from qa.data import Vocabulary
from qa.interactive import TransformerModelConfig, Generator, generator
from qa.util import get_device

import logging
logger = logging.getLogger("Generator")

def answer_with_qa_model(args: argparse.Namespace):
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
                                  saved_model_path=args.model_path)
  
  generator = Generator(config)
  
  while True:
    input_passage = input("Passage: ")
    input_question = input("Question: ")
    answer, answer_tokens, attention, _ = generator.answer(input_passage, input_question, config.max_answer_len)
    figure_path = '_'.join(input_passage[:10].split()) + ".png"
    
    print("Answer: " + answer)
    print()

def add_subparser(subparsers: argparse._SubParsersAction):
  parser = subparsers.add_parser('answer', help='Answer a passage/question pair with a trained Generative QA model')
  
  group = parser.add_argument_group('Transformer model configurations')
  group.add_argument('--hid_dims', default=256, type=int,
                       help='hidden vector dimensions')
  group.add_argument('--enc_layers', default=8, type=int,
                       help='number of encoder layers')
  group.add_argument('--dec_layers', default=8, type=int,
                       help='number of decoder layers')
  group.add_argument('--enc_heads', default=4, type=int,
                       help='number of encoder attention heads')
  group.add_argument('--dec_heads', default=4, type=int,
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
  group.add_argument('--enc_dropout', default=0.25, type=float,
                      help='encoder dropout rate')
  group.add_argument('--dec_dropout', default=0.25, type=float,
                       help='decoder dropout rate')
  
  group = parser.add_argument_group('Vocabulary and model paths')
  group.add_argument('--src_vocab', required=False,
                     help='source BPE model file path')
  group.add_argument('--trg_vocab', required=False,
                     help='target BPE model file path')
  group.add_argument('--model_path', default='transformer_qa.pt',
                      help='trained model path')
  parser.set_defaults(func=answer_with_qa_model)