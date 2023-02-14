print("Start")
import torch
from byol_pytorch import BYOL
from torchvision import models,datasets,transforms
import torch.nn as nn
import time


ROOT = "/home-mscluster/erex/research_project"
data_path = ROOT + "/NLST_dataset"
MODEL_PATH = ROOT+"/byol/saving_dir3/checkpoint.pt"
input_size=224
epochs=100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_model(model,optimizer,EPOCH,LOSS,PATH=MODEL_PATH):
    torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

def main():
    print("============ Initializing Model ... ============")

    resnet = models.resnet50(pretrained=True)

    learner = BYOL(
        resnet,
        image_size = input_size,
        hidden_layer = 'avgpool',
    )

    # ============ Making sure to use GPUs available ... ============
    learner = learner.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        learner = nn.DataParallel(learner)

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    print("============ preparing data ... ============")

    transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(data_path, transform=transform)
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # sampler=sampler,
        batch_size=64
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    print("============ Starting training ... ============") 

    best_loss = None

    since = time.time()
    for epoch in range(epochs):

        running_loss = 0.0
        for images, _ in data_loader:
            images = images.to(device)
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(data_loader.dataset)


        print('Epoch {} Loss: {:.4f}'.format(epoch, epoch_loss))
        time_elapsed = time.time() - since
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Save model if best so fat
        if best_loss is None or epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': resnet.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_loss,
                }, MODEL_PATH)
        print("================================================")

    # # saving best network is already done inside for loop
    # resnet.load_state_dict(best_model_wts) # load best model weights
    # torch.save(resnet.state_dict(), MODEL_PATH)

    print("============ Done training ! ============")
    print(" Final Loss: {} ".format(best_loss))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":
    main()
