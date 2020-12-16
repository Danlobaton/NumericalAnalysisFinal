import torch
import numpy as np
from PIL import Image
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


# CNN arch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer
        self.fc2 = nn.Linear(500, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def process_image(path_x, batch_size=20):
    my_x = np.load(path_x, allow_pickle=True)
    X = np.empty_like(my_x)
    for i in range(len(X)):
        im = Image.fromarray(np.uint8(my_x[i] * 255))
        sqrWidth = np.ceil(np.sqrt(im.size[0] * im.size[1])).astype(int)
        X[i] = im.resize((sqrWidth, sqrWidth))
    transform = transforms.Compose(
        [transforms.Scale((32, 32)),
         transforms.ToTensor()])

    tensor_x = torch.stack([transform(i) for i in X])
    return tensor_x

def train(images_path,labels_path, validation_percent, n_epochs, saved_model_name):
    np_x = np.load(images_path, allow_pickle=True)
    np_y = np.load(labels_path)

    X = np.empty_like(np_x)
    for i in range(len(X)):
        im = Image.fromarray(np.uint8(np_x[i]*255))
        sqrWidth = np.ceil(np.sqrt(im.size[0]*im.size[1])).astype(int)
        X[i] = im.resize((sqrWidth, sqrWidth))

    batch_size = 20
    transform = transforms.Compose(
                       [transforms.Scale((32,32)),
                       transforms.ToTensor()])

    transform_y = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor_x = torch.stack([transform(i) for i in X])
    tensor_y = torch.from_numpy(np_y)

    my_dataset = utils.TensorDataset(tensor_x,tensor_y)
    num_train = len(my_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_percent * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers= 0)
    valid_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
        sampler=valid_sampler, num_workers= 0)

    classes = ['a', 'b', 'c', 'd', 'h',
               'i', 'j', 'k']

    model = Net()
    print('Creating CNN ...')

    criterion = nn.CrossEntropyLoss()
    # Set stochastic gradient descent as optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    class_correct = list(0. for i in range(8))
    class_total = list(0. for i in range(8))
    valid_loss_min = np.Inf
    batch_size = 18
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        model.eval()
        for data, target in valid_loader:
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)

        
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        _,pred = torch.max(output,1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label-1] += correct[i].item()
            class_total[label-1] += 1
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))


        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss

    for i in range(8):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    torch.save(model.state_dict(), saved_model_name)

images_path = 'ClassData.npy'
labels_path = 'ClassLabels.npy'
validation_percent = 0.2
model_name = 'model.pth'
epochs = 30
#train(images_path,labels_path, validation_percent = validation_percent, n_epochs = epochs, saved_model_name = model_name)