import argparse
import os 

from data_loader import get_data_loader
from trainer import Trainer
from unbiasedbert import load_model


def main(args):
    model = load_model()
    trainer = Trainer(model)

    train_dataloader = get_data_loader('augmentation_data/train_documents.txt')
    val_dataloader = get_data_loader('augmentation_data/val_documents.txt')
    test_dataloader = get_data_loader('augmentation_data/test_documents.txt')

    trainer.train(train_dataloader, val_dataloader, debias_method=args.m, epochs=args.e, checkpoint_dir=args.checkpoint_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, help='debias method to use: active, both, orig')
    parser.add_argument('-e', type=int, help='epochs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()
    main(args)