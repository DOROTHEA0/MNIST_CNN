import argparse
from model.cnn import train


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="number of batch of each epoch")
    parser.add_argument("--num_works", type=int, default=6, help="number of thread works")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()
    train(args)
