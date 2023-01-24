import torch 
#load activation function
import torch.nn.functional as F
#load optimizer
import torch.optim as optim
# load the CIFAR10 dataset
from torchvision import datasets, transforms
# load DataLolader
from torch.utils.data import DataLoader
# define the transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,)),
])

def load_data():
    # load data using DataLoader and have a batch size of 64 from the directory "/ibex/reference/CV/CIFAR/cifar-10-batches-py"
    trainset = datasets.CIFAR10(root="/ibex/reference/CV/CIFAR/", download=False, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.CIFAR10(root="/ibex/reference/CV/CIFAR/", download=False, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)
    return trainloader, testloader

# write a function to load the data
# def load_data():
#     # download and load the training data
#     trainset = datasets.CIFAR10('data', download=True, train=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
#     # download and load the test data
#     testset = datasets.CIFAR10('data', download=True, train=False, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
#     return trainloader, testloader

# train a model
def train_model(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
    steps = 0
    # change to cuda
    model.to(device)
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every))
                running_loss = 0

# test the model
def test_model(model, testloader, criterion, device='cpu'):
    test_loss = 0
    accuracy = 0
    model.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print("Test loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test accuracy: {:.3f}".format(accuracy/len(testloader)))

# define the model fror CIFAR10
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3072, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.Linear(128, 64)
        self.fc6 = torch.nn.Linear(64, 10)
        # Define dropout, with 0.2 drop probability
        self.dropout = torch.nn.Dropout(p=0.2)
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        # output so no dropout here
        x = F.log_softmax(self.fc6(x), dim=1)
        return x



# execute the code
trainloader, testloader = load_data()
model = Classifier()
criterion = torch.nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
train_model(model, trainloader, 3, 40, criterion, optimizer, 'cuda')
test_model(model, testloader, criterion, 'cuda')
