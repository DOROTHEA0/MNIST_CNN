import torch
import model.cnn
from model.cnn import CNNTrainer, detect_image, MnistCNN
import argparse


def get_args_parser():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch size")
    parser.add_argument("--num_works", type=int, default=6, help="number of thread works")
    parser.add_argument("--test_data", type=str, default='data/kmnist-test-imgs.npz', help="test img, .npz file or .jpg, .png image file")
    parser.add_argument("--test_label", type=str, default='data/kmnist-test-labels.npz', help="test label")
    parser.add_argument("--saved_model", type=str, default='saved_models/best_model.pth', help="path to saved model")
    parser.add_argument("--device", type=str, default='cuda0', help="training device cpu or cuda0")
    return parser.parse_args()


if __name__ == '__main__':
    idx_to_result = {0: 'お', 1: 'き', 2: 'す', 3: 'つ', 4: 'な', 5: 'は', 6: 'ま', 7: 'や', 8: 'れ', 9: 'を'}
    args = get_args_parser()
    if '.npz' in args.test_data:
        trainer = CNNTrainer(args, model.cnn.MnistCNN())
        trainer.load(args.saved_model)
        print(trainer.best_acc)
        trainer.validate()
    else:
        model = MnistCNN()
        model.load_state_dict(torch.load(args.saved_model)['state_dict'])
        device = args.device
        print(idx_to_result[detect_image(args.test_data, model, device)])

