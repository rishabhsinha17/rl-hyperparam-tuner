import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from rl_tuner import RateEnv, make_agent
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    model = torchvision.models.resnet18(num_classes=10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    env = RateEnv()
    agent = make_agent(env)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        lr = agent.sample()
        for g in optimizer.param_groups:
            g['lr'] = lr
        model.train()
        imgs = 0
        start = time.time()
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()
            imgs += x.size(0)
        end = time.time()
        throughput = imgs / (end - start)
        agent.observe(loss.item(), throughput)

if __name__ == '__main__':
    main()
