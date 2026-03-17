from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import transforms as transforms
import numpy as np
import os
import argparse
import utils
import utils2
from fer import FER2013
from models.resnet_reg2 import ResNet18RegressionTwoOutputs
import pandas as pd
import torch.utils.data
import csv

# -------------------- Utils --------------------

def custom_transform(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def save_checkpoint(state, filename):
    torch.save(state, filename)

# -------------------- Train --------------------

def train(epoch, trainloader):
    net.train()
    total_loss = 0.0
    total_samples = 0

    print(f"\nEpoch: {epoch}")
    print(f"LR: {optimizer.param_groups[0]['lr']}")

    for inputs, targets in trainloader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        outputs = outputs.squeeze(1)

        # Orthogonal regularization
        diff = utils2.deconv_orth_dist(net.layer2[0].shortcut[0].weight, stride=2) + \
               utils2.deconv_orth_dist(net.layer3[0].shortcut[0].weight, stride=2) + \
               utils2.deconv_orth_dist(net.layer4[0].shortcut[0].weight, stride=2)

        diff += utils2.deconv_orth_dist(net.layer1[0].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net.layer1[1].conv1.weight, stride=1)

        diff += utils2.deconv_orth_dist(net.layer2[0].conv1.weight, stride=2)
        diff += utils2.deconv_orth_dist(net.layer2[1].conv1.weight, stride=1)

        diff += utils2.deconv_orth_dist(net.layer3[0].conv1.weight, stride=2)
        diff += utils2.deconv_orth_dist(net.layer3[1].conv1.weight, stride=1)

        diff += utils2.deconv_orth_dist(net.layer4[0].conv1.weight, stride=2)
        diff += utils2.deconv_orth_dist(net.layer4[1].conv1.weight, stride=1)

        loss = criterion(outputs, targets)
        loss = loss + 0.1 * diff

        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        total_loss += loss.item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

# -------------------- Evaluation --------------------

def evaluate(dataloader):
    net.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            outputs_avg = outputs_avg.squeeze(1)

            loss = criterion(outputs_avg, targets)

            total_loss += loss.item()
            total_samples += targets.size(0)

    return total_loss / total_samples

# -------------------- Main --------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    opt = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    total_epoch = 50
    early_stop_patience = 20

    best_loss = float('inf')
    early_stop_counter = 0

    path = "FER2013_ResNet"
    os.makedirs(path, exist_ok=True)

    # -------------------- Data --------------------

    transform_train = transforms.Compose([
        transforms.RandomCrop(44),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.TenCrop(44),
        custom_transform,
    ])

    trainset = FER2013(split='Training', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True)

    pubset = FER2013(split='PublicTest', transform=transform_test)
    publoader = torch.utils.data.DataLoader(pubset, batch_size=opt.bs)

    priset = FER2013(split='PrivateTest', transform=transform_test)
    priloader = torch.utils.data.DataLoader(priset, batch_size=opt.bs)

    # -------------------- Model --------------------

    net = ResNet18RegressionTwoOutputs()
    if use_cuda:
        net.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    prev_lr = optimizer.param_groups[0]['lr']

    checkpoint_path = os.path.join(path, 'checkpoint.pth')
    
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
    
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("No checkpoint found, starting from scratch")
        start_epoch = 0

    # -------------------- Logging --------------------

    log_file = os.path.join(path, "log.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'public_loss', 'private_loss', 'lr'])

    # -------------------- Training Loop --------------------

    for epoch in range(start_epoch, total_epoch):
        train_loss = train(epoch, trainloader)
        pub_loss = evaluate(publoader)
        pri_loss = evaluate(priloader)

        print(f"Public Loss: {pub_loss:.4f}")
        print(f"Private Loss: {pri_loss:.4f}")

        # Scheduler
        scheduler.step(pub_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if current_lr != prev_lr:
            print(f"LR reduced: {prev_lr} → {current_lr}")

        prev_lr = current_lr

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, pub_loss, pri_loss, current_lr])

        # Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_checkpoint(checkpoint, os.path.join(path, 'checkpoint.pth'))

        # Best model
        if pub_loss < best_loss:
            print("Saving best model...")
            best_loss = pub_loss
            save_checkpoint(checkpoint, os.path.join(path, 'best_model.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

    # Save last model
    torch.save(net.state_dict(), os.path.join(path, 'last_model.pth'))