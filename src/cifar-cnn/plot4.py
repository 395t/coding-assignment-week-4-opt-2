import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # Data augmentation is only done on training images
validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True) # Batch size of 100 i.e to work with 100 images at a time
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# We iter the batch of images to display
dataiter = iter(training_loader) # converting our train_dataloader to iterable so that we can iter through it. 
images, labels = dataiter.next() #going from 1st batch of 100 images to the next batch


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(vgg16)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#optimizers = {'opt_adam': opt_adam, 'opt_amsgrad': opt_amsgrad, 'opt_adamw': opt_adamw, 'opt_adagrad':opt_adagrad, 'opt_adadelta': opt_adadelta, 'opt_adadelta2': opt_adadelta2}


optimizers = ['adam', 'amsgrad', 'adamw',]
#optimizers = ['adam', 'amsgrad']

plt.title("Momentum")
plt.xlabel('Epoch')
plt.ylabel('Validation Acc')
for opt in optimizers:
    if opt == 'adam':
        ms = [0, 0.5, 0.9, 0.99]
    elif opt == 'adamw':
        ms = [0, 0.5, 0.9, 0.99]
    elif opt == 'amsgrad':
        ms = [0, 0.5, 0.9, 0.99]
    else:
        print("ERROR: Optimizer", opt, "Not found")
    plt.title("Acc vs Momentum: " + opt)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Acc')
    for m in ms:
        print("==========", opt, m, "==========")
        model = VGG().to(device)
        criterion = nn.CrossEntropyLoss() 
        epochs = 8
        params = model.parameters()
        if opt == 'adam':
            optimizer = torch.optim.Adam(params, lr=0.001, betas=(m,0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        elif opt == 'amsgrad':
            optimizer = torch.optim.Adam(params, lr=0.001, betas=(m,0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        elif opt == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=0.001, betas=(m,0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        else:
            print("ERROR: Optimizer", opt, "Not found")
        running_loss_history = []
        running_corrects_history = []
        val_running_loss_history = []
        val_running_corrects_history = []
        
        for e in range(epochs): # training our model, put input according to every batch.
          
          running_loss = 0.0
          running_corrects = 0.0
          val_running_loss = 0.0
          val_running_corrects = 0.0
          
          for inputs, labels in training_loader:
            inputs = inputs.to(device) # input to device as our model is running in mentioned device.
            labels = labels.to(device)
            outputs = model(inputs) # every batch of 100 images are put as an input.
            loss = criterion(outputs, labels) # Calc loss after each batch i/p by comparing it to actual labels. 
            
            optimizer.zero_grad() #setting the initial gradient to 0
            loss.backward() # backpropagating the loss
            optimizer.step() # updating the weights and bias values for every single step.
            
            _, preds = torch.max(outputs, 1) # taking the highest value of prediction.
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data) # calculating te accuracy by taking the sum of all the correct predictions in a batch.
        
          else:
            with torch.no_grad(): # we do not need gradient for validation.
              for val_inputs, val_labels in validation_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                
                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)
              
            epoch_loss = running_loss/len(training_loader) # loss per epoch
            epoch_acc = running_corrects.float()/ len(training_loader) # accuracy per epoch
            running_loss_history.append(epoch_loss) # appending for displaying 
            running_corrects_history.append(epoch_acc.cpu())
            
            val_epoch_loss = val_running_loss/len(validation_loader)
            val_epoch_acc = val_running_corrects.float()/ len(validation_loader)
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc.cpu())
            print('epoch :', (e+1))
            print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
            print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))
        label = str(m)
        plt.plot(val_running_corrects_history, label=label)
    
    plt.legend()
    plt.show()