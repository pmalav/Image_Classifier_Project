import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', action = 'store', default = 'ImageClassifier/flowers', help = 'Directory containing all the images' )
    parser.add_argument('--arch', action = 'store', type = str, default = 'vgg16', help = 'Type of Architecture to be used')
    parser.add_argument('--save_dir', type = str, action = 'store', default = 'checkpoint.pth', help = 'Model Checkpoint Directory')
    parser.add_argument('--learning_rate', action = 'store', type = float, default = 0.0005, help = 'Decide the learning rate')
    parser.add_argument('--hidden_units_1', action = 'store', type = int, default = 600, help = 'First layer of hidden units')
    parser.add_argument('--hidden_units_2', action = 'store', type = int, default = 400, help = 'Second layer of hidden units')
    parser.add_argument('--hidden_units_3', action = 'store', type = int, default = 200, help = 'Third layer of hidden units')
    parser.add_argument('--epochs', action = 'store', type = int, default = 10, help = 'Number of epochs')
    parser.add_argument('--gpu', action = 'store', default = 'cuda', help = 'Type of device to be used')

    in_arg = parser.parse_args()
    
    data_dir = in_arg.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=train_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    device = torch.device(in_arg.gpu)

    if in_arg.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    if in_arg.arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    if in_arg.arch == 'alexnet': 
        model = models.alexnet(pretrained=True)
    else:
        print("Training will proceed with the default network vgg16. Either you did not specify the network for training or the network specified by you is not available for training the model.")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088, in_arg.hidden_units_1),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_arg.hidden_units_1, in_arg.hidden_units_2),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_arg.hidden_units_2, in_arg.hidden_units_3),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_arg.hidden_units_3, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), in_arg.learning_rate)

    model.to(device);

    epochs = in_arg.epochs
    steps = 0
    running_train_loss = 0

    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            if steps % print_every == 0:
                running_valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        running_valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_train_loss/print_every:.3f}.. "
                      f"Validation loss: {running_valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

                running_train_loss = 0
                model.train()

    print("Training Completed!")

    model.class_to_idx = train_data.class_to_idx


    checkpoint = {
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'model_class_to_index': model.class_to_idx,
        'hidden_units_1': in_arg.hidden_units_1,
        'hidden_units_2': in_arg.hidden_units_2,
        'hidden_units_3': in_arg.hidden_units_3,
        'learning_rate': in_arg.learning_rate,
        'arch': in_arg.arch,
        'save_dir': in_arg.save_dir
    }

    torch.save(checkpoint, in_arg.save_dir)
