import argparse
from model.cnn import CNNTrainer


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="number of batch of each epoch")
    parser.add_argument("--num_works", type=int, default=6, help="number of thread works")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--device", type=str, default='cuda0', help="training device cpu or cuda0")
    parser.add_argument("--train_data", type=str, default='data/kmnist-train-imgs.npz', help="train img")
    parser.add_argument("--train_label", type=str, default='data/kmnist-train-labels.npz', help="train label")
    parser.add_argument("--val_data", type=str, default='data/kmnist-val-imgs.npz', help="val img")
    parser.add_argument("--val_label", type=str, default='data/kmnist-val-labels.npz', help="val label")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()
    trainer = CNNTrainer(args)
    trainer.train()
