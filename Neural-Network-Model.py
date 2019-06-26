import shutil, os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from data_loader_tester import *
from gcommand_loader import GCommandLoader, find_classes

# Hyperparameters
num_epochs = 10
num_classes = 30
batch_size = 100
learning_rate = .0001

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 60, kernel_size=3, stride=2, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
             nn.Conv2d(60, 100, kernel_size=6, stride=2, padding=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=[2,5], stride=2))
             #out = 12 * 13 * 100
             
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(20 * 11 * 100 , 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 30)

    def forward(self, x):
#        print("X: ", x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def validation_model(self, validation_set):
#        print('computing validation')
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for examples, labels in validation_set:
                examples, labels = examples.to("cuda"), labels.to("cuda")
                outputs = self(examples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('valid Test Accuracy of the model on the examples: {} %'.format((correct / total) * 100))

#    def test_prediction(self, test_loader):
#        f = open("test_y", "w")
#    for x in test_loader:
#        x = np.reshape(x, (1, 784))
#        out = forward(x, parameters, sigmoid)
#        y_hat = out['y_hat']
#        f.write(str(y_hat.argmax()) + '\n')
#    f.close()
#
#
#




def train(model, criterion , optimizer, train_loader, device, m ):
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Run the forward pass
            outputs = model(images)
            loss = criterion(m(outputs), labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))
        print("start backprop with epoch:{}, loss : {}, acc :{} ".format( epoch + 1, loss.item(), (correct / total) * 100))


def predict(input, model, m, device, original):
    f = open("test_y", "w")
    for i, (images, labels) in enumerate(input):
        name = original.spects[i][0]
        name = name.split("/")[-1]
        images = images.to(device)
        result = model(images)
        result = m(result)
        indice = result.argmax(1)
        #indice = indice[0]
        phrase = name + ", " + str(indice[0].item()) + "\n"
        f.write(phrase)
    f.close()

def moveAllFilesinDir(srcDir, dstDir):
    # Check if both the are directories
    if os.path.isdir(srcDir) and os.path.isdir(dstDir) :
        # Iterate over all the files in source directory
        for filePath in glob.glob(srcDir + '\*'):
            # Move each file to destination Directory
            shutil.move(filePath, dstDir);
    else:
        print("srcDir & dstDir should be Directories")


def main():

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    
#    sourceDir = './data/test'
##    destDir =  './data/test_folder/'
##    os.makedirs(destDir)
#    destDir =  './data/test_folder/test'
#    os.makedirs(destDir)
#    moveAllFilesinDir(sourceDir,destDir)

    dataSetTest = GCommandLoader("./data/test_folder")
#    names = os.listdir("./data/test_folder/test")
#    names.sort()
    dataSetTrain = GCommandLoader("./data/train")
    dataSetValid = GCommandLoader("./data/valid")

    print(" test size ", len(dataSetTest))
    print(" train size ", len(dataSetTrain))
    print(" valid size ", len(dataSetValid))

    # Data Path
    # Need to store in a location for the trained model parameters once training is complete.
    
    test_loader = torch.utils.data.DataLoader(
        dataSetTest, batch_size=1, shuffle=False)    # datatest_set remplace par dataSetTest
    
    train_loader = torch.utils.data.DataLoader(
        dataSetTrain, batch_size=batch_size, shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader(
        dataSetValid, batch_size=batch_size, shuffle=True)




    model = ConvNet().to(device)

    # Loss and optimizer
    m = nn.LogSoftmax(dim=1)
    criterion =  nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    #train
    train(model, criterion, optimizer, train_loader, device, m)

    # validation
    model.validation_model(valid_loader)

#    dataset = torch.tensor(dataset)
    test_predictions = predict(test_loader, model, m, device, dataSetTest)

main()

