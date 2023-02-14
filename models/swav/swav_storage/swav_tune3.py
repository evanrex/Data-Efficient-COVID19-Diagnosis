from __future__ import print_function 
from __future__ import division

from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import src.resnet50 as resnet_models


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


ROOT = "/home-mscluster/erex/research_project"

data_dir = ROOT + "/Covidx-CT"
num_classes = 2
batch_size = 32 # Batch size for training (change depending on how much memory you have)
num_epochs = 15 # Number of epochs to train for 
feature_extract = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRETRAINED_PATH = ROOT+ "/swav/saving_dir3/checkpoints/ckp-99.pth"

MODEL_PATH = ROOT + "/swav/saving_dir3/swav_model.pt"


def load_data(input_size):
    '''
    Arguments: 
        input_size
    Returns: 
        a dictionary of data loaders.
    
    '''

    # train_dataloader = ImageFolder(root=TRAIN_PATH)
    # valid_dataloader = ImageFolder(root=VALID_PATH)
    # test_dataloader  = ImageFolder(root=TEST_PATH)
    # image_datasets = {
    #     'train':train_dataloader,
    #     'validation':valid_dataloader,
    #     'test':test_dataloader
    #     }

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation','test']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'validation','test']}
    return dataloaders_dict

def save_model(model,optimizer,EPOCH,LOSS,PATH=MODEL_PATH):
    torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels = 2, use_bn=False):
        super(RegLog, self).__init__()
        self.bn = None

        s = 8192
        self.av_pool = nn.AvgPool2d(6, stride=1)
        if use_bn:
            self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # average pool the final feature map
        x = self.av_pool(x)

        # optional BN
        if self.bn is not None:
            x = self.bn(x)

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def load_saved_model(PATH=MODEL_PATH):

    '''
    Returns:
        model_ft: Pytorch model object,
        input_size: input size for model_ft
    '''

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    
    model_ft = resnet_models.__dict__["resnet50"](output_dim=0, eval_mode=True)
    
    checkpoint = torch.load(PATH)
    state_dict = checkpoint["state_dict"]

    # remove prefixe "module."
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    for k, v in model_ft.state_dict().items():
        if k not in list(state_dict):
            print('key "{}" could not be found in provided state dict'.format(k))
        elif state_dict[k].shape != v.shape:
            print('key "{}" is of different shape in model and provided state dict'.format(k))
            state_dict[k] = v

    msg = model_ft.load_state_dict(state_dict, strict=False)
    print(("Load pretrained model with msg: {}".format(msg)))

    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = 8192 #model_ft.fc.in_features # hard coding it because swav is weird
    # linear_model = nn.Linear(num_ftrs, num_classes)
    linear_model = RegLog()
    input_size = 224

    return (model_ft, linear_model), input_size

def train_model(swavResNet, linear_model,dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(linear_model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                linear_model.train()  # Set model to training mode
            else:
                linear_model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    input_extracted = swavResNet(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = linear_model(input_extracted)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = linear_model(input_extracted)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(linear_model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': linear_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    }, MODEL_PATH)
            if phase == 'validation':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    linear_model.load_state_dict(best_model_wts)
    return linear_model, val_acc_history

def initialize_model(use_pretrained=True,feature_extract=True,num_classes=2):
    '''
    Returns:
        model_ft: Pytorch model object,
        input_size: input size for model_ft
    '''
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    
    model_ft = models.resnet50(weights= "IMAGENET1K_V2")
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size

def initialize_optimizer(model_ft):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer_ft

def main():
    '''
    Notes regarding Cluster requirements:
        This code is parallelised
        Saves a checkpoint of best weights every epoch
        Outputs regular updates

    Useful resources:
        https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    '''
    (model_ft,linear_model), input_size = load_saved_model(PATH=PRETRAINED_PATH)
    dataloaders_dict = load_data(input_size)
    model_ft = model_ft.to(device)
    linear_model = linear_model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = nn.DataParallel(model_ft)
        linear_model = nn.DataParallel(linear_model)
    
    optimizer_ft = initialize_optimizer(linear_model)
    criterion = nn.CrossEntropyLoss()
    # Train and evaluate
    print("Starting training...")
    linear_model, hist = train_model(model_ft, linear_model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    

if __name__ == "__main__":
    main()
