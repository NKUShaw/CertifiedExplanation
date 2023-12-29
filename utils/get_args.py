import argparse

def get_arg():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-1, help='Learning rate', dest='lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay for SGD')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--clip_grads', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--frac_escape', type=float, default=0.2, help='fraction of data escaped')
    parser.add_argument('--num_escape', type=int, default=145)
    parser.add_argument('--noise_scale', type=float, default=1.0)


    return parser.parse_args()