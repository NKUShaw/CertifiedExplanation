import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from utils import autograd_hacks
import tqdm
import logging
# import wandb
from models.model import *
from torch import optim
from utils.get_args import get_arg
from utils.get_datasets import get_dataset
from torch.utils.data import DataLoader
from pathlib import Path

# wandb.init(mode='offline')


def test(model, device, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # Do not compute gradient during testing
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# def add_gaussian_noise(args, input):
#     noise = torch.randn_like(input) * args.noise_scale
#     return input + noise
#
#
# def predict_with_randomized_smoothing(args, model, image, num_samples):
#     # Expand the single image to a batch
#     image_batch = image.unsqueeze(0).repeat(num_samples, 1, 1, 1)
#     # Add Gaussian noise
#     noisy_images = add_gaussian_noise(args, image_batch)
#
#     # Predict using the model
#     predictions = model(noisy_images)
#
#     # Get the predicted classes for each noisy image
#     _, predicted_classes = predictions.max(1)
#
#     # Return the most frequently predicted class
#     return torch.bincount(predicted_classes).argmax().item()

if __name__ == '__main__':
    args = get_arg()
    device = 'cuda'
    dir_checkpoint = Path(f'./checkpoints/')

    # 1. Dataloader
    train_dataset, test_dataset = get_dataset(args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Initialize logging
    # experiment = wandb.init(project='UAI', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, momentum=args.momentum,
    #          optimizer=args.optimizer, sigma=args.sigma, )
    # )
    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Optimizer:       {args.optimizer}
        Sigma:           {args.sigma}
        Momentum:        {args.momentum}
        Weight decay:    {args.weight_decay}
    ''')
    # 3. Obtain model
    if args.model == 'cnn' and args.dataset == 'cifar':
        model = CNNCifar10()
    elif args.model == 'cnn' and args.dataset == 'fashion':
        model = CNNFashion()
    elif args.model == 'mlp' and args.dataset == 'mnist':
        model = MLPMnist()
    else:
        print("还没更新！")
    model = model.cuda()
    autograd_hacks.add_hooks(model)
    # 4. Set up the optimizer, the loss, the learning rate scheduler
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('我还没写')

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    global_step = 0
    best_accuracy = 0.0
    print('==> Building model..')

    # 5. Traning
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch} / {args.epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                autograd_hacks.clear_backprops(model)
                loss.backward()
                autograd_hacks.compute_grad1(model)
                optimizer.step()
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix({"Loss": epoch_loss / (batch_idx + 1)})
                pbar.update(1)
        # Randomized
        for index, param in enumerate(model.parameters()):
            if not hasattr(param, 'grad1'):
                assert (param.grad == None)
                continue
            grads_per_sample = param.grad1
            grads_per_sample = grads_per_sample.reshape(grads_per_sample.shape[0], -1)
            w0 = grads_per_sample.mean(dim=0)
            ############################################################################
            des, _ = torch.sort(grads_per_sample, dim=0, descending=True)
            e11 = des[args.num_escape:]
            e1 = e11.mean(dim=0)
            e21 = des[:-args.num_escape]
            e2 = e21.mean(dim=0)
            e = torch.stack((e1, e2), dim=0)
            e = torch.abs(torch.sub(w0, e))
            e, _ = torch.max(e, dim=0)
            ############################################################################
            milestone = args.sigma * args.sigma / 2
            m = torch.distributions.normal.Normal(0, args.sigma * args.sigma)
            g_x = torch.zeros(w0.shape[0], 3).to(device)
            g_x[:, 0] = 1 - m.cdf(torch.sub(milestone, w0))
            g_x[:, 1] = m.cdf(torch.sub(milestone, w0)) - m.cdf(torch.sub(-milestone, w0))
            g_x[:, 2] = m.cdf(torch.sub(-milestone, w0))
            ############################################################################
            g_x_p, _ = torch.sort(g_x, dim=1, descending=True)
            g_x_p = g_x_p[:, :2]
            ############################################################################
            radius = (args.sigma / 2 * (m.icdf(g_x_p[:, 0]) - m.icdf(g_x_p[:, 1])))
            ############################################################################
            failed = torch.gt(e, radius)
            ############################################################################
            g_x[failed == True, 0] = 0.0
            g_x[failed == True, 1] = 1.0
            g_x[failed == True, 2] = 0.0
            ############################################################################
            v_x = torch.argmax(g_x, dim=1)
            v_x = F.one_hot(v_x, num_classes=3)
            ############################################################################
            v_x = v_x.float()
            v_x[:, 0] = torch.mul(v_x[:, 0], args.gamma)
            v_x[:, 1] = torch.mul(v_x[:, 1], 0.0)
            v_x[:, 2] = torch.mul(v_x[:, 2], -args.gamma)
            v_x = torch.sum(v_x, dim=1).float()
            ############################################################################
            v_x = v_x.reshape(param.grad.shape).to(device)
            ############################################################################
            param.grad.data = v_x
        scheduler.step()
        avg_loss, accuracy = test(model=model, device='cuda', test_loader=test_loader, criterion=criterion)
        print('------Test Result:------')
        print('Avg_loss = {}, Accuracy = {}'.format(avg_loss, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            state_dict = model.state_dict()
            best_model_path = str(
                dir_checkpoint/ '{}'.format(args.dataset) / '{}'.format(args.sigma) / 'best_model_{}.pth'.format(args.dataset))
            torch.save(state_dict, best_model_path)
            logging.info(f'Best model (accuracy: {best_accuracy:.2f}%) saved to {best_model_path}')
        # Saved model's weight as pth file
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict,
                   str(dir_checkpoint / '{}'.format(args.dataset) / '{}'.format(args.sigma) / 'checkpoint_{}_epoch{}.pth'.format(args.dataset,
                                                                                                       epoch)))
        logging.info(f'Checkpoint {epoch} saved!')
        print('Best Accuracy = {}'.format(best_accuracy))
