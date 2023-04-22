import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from expert_dataset import ExpertDataset
from models.cilrs import CILRS
import matplotlib.pyplot as plt
from torchvision import transforms
# import wandb
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
global_step = 0

def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here
    # global global_step
    model.eval()
    running_loss = 0 
    crit = nn.L1Loss()
    # crit = nn.MSELoss()
    with torch.no_grad():
        for measurements,image in dataloader:
            image = image.to(device)
            speed = measurements['speed'].to(device)
            command = measurements['command'].to(device) # high level commands
            throttle = measurements['throttle'].to(device)
            brake = measurements['brake'].to(device)
            steer = measurements['steer'].to(device)

            preds_control, pred_speed=model.forward(image, command, speed)
            loss1 = crit(pred_speed.squeeze(),speed)
            loss2 = crit(preds_control[:,0],throttle)
            loss3 = crit(preds_control[:,1],brake)
            loss4 = crit(preds_control[:,2],steer)
            loss = loss1 + loss2 + loss3 + loss4
            running_loss += loss.item()
            # wandb.log({'val/Speed_loss': loss1.mean().item(),
            #        'val/throttle_loss': loss2.mean().item(),
            #        'val/brake_loss': loss3.mean().item(),
            #        'val/steer_loss': loss4.mean().item(),
            #        'val/total_loss': loss.mean().item(),
            #      },
            #      step=global_step )
            
    return running_loss/len(dataloader)


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    global global_step
    running_loss = 0 
    optimizer = optim.Adam(model.parameters(), lr =0.001)
    crit = nn.L1Loss()
    # crit = nn.MSELoss()
    for measurements,image in dataloader:
        image = image.to(device)
    
        speed = measurements['speed'].to(device)
        command = measurements['command'].to(device) # high level commands
        throttle = measurements['throttle'].to(device)
        brake = measurements['brake'].to(device)
        steer = measurements['steer'].to(device)

        optimizer.zero_grad()
        preds_control, pred_speed=model.forward(image, command, speed)
        loss1 = crit(pred_speed.squeeze(),speed.float()) # the input shouls have float type for L2 loss
        loss2 = crit(preds_control[:,0],throttle.float())
        loss3 = crit(preds_control[:,1],brake.float())
        loss4 = crit(preds_control[:,2],steer.float())
        loss = 0.25 * loss1 + loss2 + loss3 + loss4 # so that speed will be in ranges
        loss.backward() 
        running_loss += loss.item()
        optimizer.step()

        # wandb.log({'Train/Speed_loss': loss1.mean().item(),
        #            'Train/throttle_loss': loss2.mean().item(),
        #            'Train/brake_loss': loss3.mean().item(),
        #            'Train/steer_loss': loss4.mean().item(),
        #            'Train/total_loss': loss.mean().item(),
        #          },
        #          step = global_step)
        global_step += 1
    return running_loss/len(dataloader)



def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    plt.plot(torch.tensor(train_loss).detach().cpu().numpy(),'r')
    plt.plot(torch.tensor(val_loss).detach().cpu().numpy(), 'b')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Losses-CILRS-L1')
    plt.legend(['Train loss', 'Test loss'])
    plt.savefig('Losses-cilrs-L1.png')


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    # wandb.init(project='cvad-hw1', name='CILRS-processed_data-L1_Loss')
    train_root = "/userfiles/ssafadoust20/expert_data/train" # inside there is a measurement and rgb
    val_root = "/userfiles/ssafadoust20/expert_data/val"
    model = CILRS().to(device)
    T = transforms.Compose([
    transforms.Resize((224, 224)),   #
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
    train_dataset = ExpertDataset(train_root, T)
    val_dataset = ExpertDataset(val_root, T)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 128
    save_path = "cilrs_model-L1-loss.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
        print('Epoch: ', i, 'Train loss is: ', train(model, train_loader), 'VAl loss is: ', validate(model, val_loader) )
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
