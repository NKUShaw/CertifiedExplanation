from torchvision import transforms, datasets


def get_dataset(args):

    if args.dataset == 'cifar':
        data_dir = './data/{}'.format(args.dataset)
        apply_transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Normalize((0.5,), (0.5,))])
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
    elif args.dataset == 'cifar100':
        data_dir = './data/{}'.format(args.dataset)
        apply_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                              transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                             transform=apply_transform)
    elif args.dataset == 'caltech101':
        data_dir = './data/{}'.format(args.dataset)
        apply_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((256, 256)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        train_dataset = datasets.Caltech101(data_dir, train=True, transform=apply_transform, download=True)
        test_dataset = datasets.Caltech101(data_dir, train=False, transform=apply_transform, download=True)

    else:
        print('暂时没加这个数据集')

    return train_dataset, test_dataset

