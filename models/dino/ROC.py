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
import numpy as np
import torch.nn.functional as F

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

ROOT = "/home-mscluster/erex/research_project"

data_dir = ROOT + "/Covidx-CT"
num_classes = 2
batch_size = 32 # Batch size for training (change depending on how much memory you have)
num_epochs = 15 # Number of epochs to train for 
feature_extract = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASELINE_PATH = ROOT + "/baseline/baseline_chkpt/model.pt"

SAVING_PATH = ROOT+"/dino/ROC_csvs/"

DINO1_PATH = ROOT+"/dino/saving_dir/" + "dino_model.pt"
DINO2_PATH = ROOT+"/dino/saving_dir2/" + "dino_model.pt"
DINO3_PATH = ROOT+"/dino/saving_dir3/" + "dino_model.pt"


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

# def save_model(model,optimizer,EPOCH,LOSS,PATH=MODEL_PATH):
#     torch.save({
#             'epoch': EPOCH,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
#             }, PATH)

def load_saved_model_dino(PATH):

    '''
    Returns:
        model_ft: Pytorch model object,
        input_size: input size for model_ft
    '''

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    
    model_ft = models.resnet50()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    checkpoint = torch.load(PATH)
    state_dict = checkpoint['model_state_dict']

    # # remove `module.` prefix
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # # remove `backbone.` prefix induced by multicrop wrapper
    # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    
    msg = model_ft.load_state_dict(state_dict, strict=False)
    print(' DINO Pretrained weights found at {} and loaded with msg: {}'.format(PATH, msg))

    input_size = 224

    return model_ft, input_size


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

def load_saved_model(PATH):

    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False

    model = models.resnet50(weights= "IMAGENET1K_V2")
    set_parameter_requires_grad(model)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    checkpoint = torch.load(PATH)

    state_dict = checkpoint['model_state_dict']

    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    msg = model.load_state_dict(state_dict)
    print('Baseline Pretrained weights found at {} and loaded with msg: {}'.format(PATH, msg))
    return model,input_size


def eval_models(baseline,dino1,dino2,dino3, dataloaders):
    # Each epoch has a training and validation phase

    baseline.eval()   # Set model to evaluate mode
    dino1.eval()   # Set model to evaluate mode
    dino2.eval()   # Set model to evaluate mode
    dino3.eval()   # Set model to evaluate mode


    phase = 'test'

    # Iterate over data.
    y_true = np.array([])
    y_scores_baseline = np.array([])
    y_scores1 = np.array([])
    y_scores2 = np.array([])
    y_scores3 = np.array([])

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward

        with torch.set_grad_enabled(False):

            
            outputs_baseline = baseline(inputs)
            probabilities_baseline = F.softmax(outputs_baseline, dim=1)[:, 1]
            y_score__baseline = probabilities_baseline.detach().cpu().numpy()
            y_scores_baseline = np.append(y_scores_baseline,y_score__baseline)

            outputs1 = dino1(inputs)
            probabilities1 = F.softmax(outputs1, dim=1)[:, 1]
            y_score1 = probabilities1.detach().cpu().numpy()
            y_scores1 = np.append(y_scores1,y_score1)

            outputs2 = dino2(inputs)
            probabilities2 = F.softmax(outputs2, dim=1)[:, 1]
            y_score2 = probabilities2.detach().cpu().numpy()
            y_scores2 = np.append(y_scores2,y_score2)

            outputs3 = dino3(inputs)
            probabilities3 = F.softmax(outputs3, dim=1)[:, 1]
            y_score3 = probabilities3.detach().cpu().numpy()
            y_scores3 = np.append(y_scores3,y_score3)

            y_labels = labels.detach().cpu().numpy()
            y_true= np.append(y_true,y_labels)

        np.savetxt(SAVING_PATH+"true_output.csv", y_true, delimiter=",")
        np.savetxt(SAVING_PATH+"baseline_output.csv", y_scores_baseline, delimiter=",")
        np.savetxt(SAVING_PATH+"dino1_output.csv", y_scores1, delimiter=",")
        np.savetxt(SAVING_PATH+"dino2_output.csv", y_scores2, delimiter=",")
        np.savetxt(SAVING_PATH+"dino3_output.csv", y_scores3, delimiter=",")

    print("Done!")


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
    baseline, input_size = load_saved_model(PATH=BASELINE_PATH)
    dino1, input_size = load_saved_model_dino(PATH=DINO1_PATH)
    dino2, input_size = load_saved_model_dino(PATH=DINO2_PATH)
    dino3, input_size = load_saved_model_dino(PATH=DINO3_PATH)

    print("Loaded models!")

    dataloaders_dict = load_data(input_size)
    print("Loaded data!")

    baseline = baseline.to(device)
    dino1 = dino1.to(device)
    dino2 = dino2.to(device)
    dino3 = dino3.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        baseline = nn.DataParallel(baseline)
        dino1 = nn.DataParallel(dino1)
        dino2 = nn.DataParallel(dino2)
        dino3 = nn.DataParallel(dino3)
    
    # Evaluate
    print("Starting Evaluation ...")
    eval_models(baseline,dino1,dino2,dino3,dataloaders_dict)




if __name__ == "__main__":
    main()
