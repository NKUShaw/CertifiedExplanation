from torchvision import transforms, datasets


def get_dataset(args):

    if args.dataset == 'cifar':
        data_dir = './data/{}'.format(args.dataset)
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

    elif args.dataset == 'fashion':
        data_dir = './data/{}'.format(args.dataset)
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                             transform=apply_transform)

    elif args.dataset == 'mnist':
        data_dir = './data/{}'.format(args.dataset)
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                             transform=apply_transform)

    else:
        print('暂时没加这个数据集')

    return train_dataset, test_dataset

