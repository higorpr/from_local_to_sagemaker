import torch
from torchvision import datasets, transforms
from torch import nn, optim

def train(model, train_loader, cost, optimizer, epoch):
    model.train()
    for e in range(epoch):
        running_loss = 0
        correct = 0
        for data, target in train_loader:
            data = data.view(data.shape[0], -1) # Reshaping data
            optimizer.zero_grad() # Resetting gradient for new batch
            pred = model(data) # Making predictions
            loss = cost(pred, target) # Calculatiing batch loss
            running_loss += loss # Calculating cumulated loss
            loss.backward() # Recalculating weights
            optimizer.step() # Updating calculated weights
            pred = pred.argmax(dim=1, keepdim=True) # Extracts classes with biggest probabilities
            correct += pred.eq(target.view_as(pred)).sum().item() # Compares predictions with labels
            print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")

def test(model, test_loader):
    model.eval()
    #TODO: Add code here to test the accuracy of your model
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # Extracts classes with biggest probabilities
            correct += pred.eq(target.view_as(pred)).sum().item() # Compares predictions with labels
            print(f"Test Set: Accuracy {(correct/len(test_loader.dataset))} = {100*(correct/len(test_loader.dataset))}")

def create_model():
    input_size = 784
    output_size = 10
    model = nn.Sequential(
        nn.Linear(input_size,128),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,output_size),
        nn.LogSoftmax(dim=1),)
    return model

#TODO: Create your Data Transforms
training_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

# Setting Hyperparameters:
batch_size = 64
epoch = 3

#TODO: Download and create loaders for your data

trainset = datasets.MNIST('data/', download=True, train=True, transform=training_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.MNIST('data/', download=True, train=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

model=create_model()

cost = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train(model, train_loader, cost, optimizer, epoch)
test(model, test_loader)
