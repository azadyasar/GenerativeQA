import argparse
from qa import (train_model)

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog='qa',
    description='Generative Question Answering'
  )
  subparsers = parser.add_subparsers(dest='subcommands')
  
  train_model.add_subparser(subparsers)
  
  args = parser.parse_args()
  args.func(args)