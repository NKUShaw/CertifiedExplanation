import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

import models.capsulecifar
import tqdm
import logging
# import wandb
from models.model import *
from torch import optim
from utils.get_args import get_arg
from utils.get_datasets import get_dataset
from utils.get_losses import CapsuleLoss, OrthogonalProjectionLoss
from torch.utils.data import DataLoader
from pathlib import Path
# wandb.init(mode='offline')

num_classes = 10
def test_capsule(model, device, test_loader, criterion):
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        # Add channels = 1
        images = images.to(device)
        # Categogrical encoding
        labels = torch.eye(num_classes).index_select(dim=0, index=labels).to(device)
        logits, reconstructions, primary_caps_output, digit_caps_output, c, b = model(images)
        pred_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(pred_labels == torch.argmax(labels, dim=1)).item()
        total += len(labels)

    print('Accuracy: {}'.format(correct / total))
    return correct / total

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
    # model = models.capsule_argmax.CapsNet(device=device)
    model = models.capsulecifar.CapsNet(device=device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)
    # 4. Set up the optimizer, the loss, the learning rate scheduler
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('我还没写')

    criterion = CapsuleLoss()
    # criterion2 = OrthogonalProjectionLoss(gamma=2.0)
    op_weight = 1.0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    global_step = 0
    best_accuracy = 0.0
    print('==> Building model..')

    # 5. Traning
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        total_m_loss = 0.0
        total_r_loss = 0.0
        # total_o_loss = 0.0
        with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch} / {args.epochs}", unit="batch") as pbar:
            batch_id = 1
            correct, total, total_loss = 0, 0, 0.
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                labels_op = labels
                optimizer.zero_grad()
                eye = torch.eye(num_classes).to(device)
                labels = eye.index_select(dim=0, index=labels)
                logits, reconstruction, primary_caps_output, digit_caps_output, c, b = model(inputs)
                loss = criterion(inputs, labels, logits, reconstruction)
                # labels_op = labels_op.view(-1)
                primary_caps_output_op = primary_caps_output.view(args.batch_size, -1)
                # loss_op = criterion2(primary_caps_output_op, labels_op)
                margin_loss = criterion.margin_loss.item()
                reconstruction_loss = criterion.reconstruction_loss.item()
                correct += torch.sum(torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()
                total += len(labels)
                accuracy = correct / total
                total_loss += loss # + op_weight * loss_op
                total_m_loss += margin_loss
                total_r_loss += reconstruction_loss
                # total_o_loss += loss_op
                loss.backward()
                optimizer.step()
                batch_id += 1
                pbar.set_postfix({
                    "Margin Loss": total_m_loss / (batch_idx + 1),
                    "Reconstruction Loss": total_r_loss / (batch_idx + 1) #"OPLoss": total_o_loss / (batch_idx + 1)
                })
                pbar.update(1)
        scheduler.step()
        print('------Test Result:------')
        test_accuracy = test_capsule(model=model, device=device, test_loader=test_loader, criterion=criterion)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            state_dict = model.state_dict()
            best_model_path = str(
                dir_checkpoint/ '{}'.format('Capsule') / '{}'.format(args.dataset) / 'common' / 'best_model_{}.pth'.format(args.dataset))
            torch.save(state_dict, best_model_path)
            logging.info(f'Best model (accuracy: {best_accuracy:.2f}%) saved to {best_model_path}')
        # Saved model's weight as pth file
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict,
                   str(dir_checkpoint / '{}'.format('Capsule') / '{}'.format(args.dataset) / 'common' / 'checkpoint_{}_epoch{}.pth'.format(args.dataset,
                                                                                                       epoch)))
        logging.info(f'Checkpoint {epoch} saved!')
        print('Best Accuracy = {}'.format(best_accuracy))
