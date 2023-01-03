import argparse
import sys

import torch
import click
from torch import nn

from data import mnist
from model import MyAwesomeModel
from torchvision import transforms
from torch.utils.data import TensorDataset
from torch.optim import Adam


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.Lambda(lambda x: torch.flatten(torch.swapdims(x, 0, 1), start_dim=1)),
                                    transforms.Lambda(lambda x: x.to(torch.float32))])

    trainset = TensorDataset(transform(train_set[0]), torch.Tensor(train_set[1]))

    # Download and load the training data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    epochs = 30

    for e in range(epochs):
        running_loss = 0
        print(f"Begin epoch: #{e}")
        for images, labels in trainloader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels.to(torch.long))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Loss={running_loss}")

    torch.save(model.state_dict(), 'checkpoint.pth')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    _, test_set = mnist()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.Lambda(lambda x: torch.flatten(torch.swapdims(x, 0, 1), start_dim=1)),
                                    transforms.Lambda(lambda x: x.to(torch.float32))])

    testset = TensorDataset(transform(test_set[0]), torch.Tensor(test_set[1]))

    # Download and load the training data
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    model.eval()
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            probs = model.forward(images)

            top_p, top_class = probs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    accuracy = accuracy / len(testloader)
    print(f'Accuracy: {accuracy * 100}%')



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
    