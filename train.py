import argparse
from model.cnn import CNNTrainer, MnistCNN, MyMnistCNN


def get_args_parser() -> argparse.ArgumentParser():
    """
    Training parameters
    :return: argparse parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch of each epoch")
    parser.add_argument("--num_works", type=int, default=6, help="number of thread works")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--device", type=str, default='cuda0', help="training device cpu or cuda0")
    parser.add_argument("--train_data", type=str, default='data/kmnist-train-imgs.npz', help="train img")
    parser.add_argument("--train_label", type=str, default='data/kmnist-train-labels.npz', help="train label")
    parser.add_argument("--val_data", type=str, default='data/kmnist-val-imgs.npz', help="val img")
    parser.add_argument("--val_label", type=str, default='data/kmnist-val-labels.npz', help="val label")
    parser.add_argument("--use_l2", type=bool, default=False, help="use l2 regularization")
    parser.add_argument("--use_lr_scheduler", type=bool, default=False, help="use learning rate scheduler")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()
    trainer = CNNTrainer(args, MnistCNN())
    # trainer = CNNTrainer(args, MyMnistCNN())
    trainer.train()


