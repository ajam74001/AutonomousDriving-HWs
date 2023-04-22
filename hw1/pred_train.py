import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import wandb
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
global_step = 0
def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here
    
    crit1 = nn.L1Loss()
    crit2 = nn.BCELoss()
    model.eval()
    running_loss= 0 
    with torch.no_grad():
        for measurements,images in dataloader:
            images= images.to(device)
            preds=model.forward(images) # going to be a dictionary 
            loss1 = crit1(preds["lane_dist"],measurements["lane_dist"].to(device))    # can i pass a 
            loss2 = crit1(preds["route_angle"],measurements["route_angle"].to(device))
            loss3 = crit1(preds["tl_dist"],measurements["tl_dist"].to(device))
            loss4 = crit2(preds["tl_state"],measurements["tl_state"].float().to(device))
            loss = loss1 + loss2 + loss3 + loss4    
            running_loss += loss.item()
            wandb.log({'Val/lane_dist_loss': loss1.mean().item(),
                   'Val/route_angle_loss': loss2.mean().item(),
                   'Val/tl_dist_loss': loss3.mean().item(),
                   'Val/tl_state_loss': loss4.mean().item(),
                   'Val/totall_loss': loss.mean().item(),
                   "step" : global_step

                 })
    return running_loss/len(dataloader)



def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    global global_step
    optimizer = optim.Adam(model.parameters(), lr =0.001)
    crit1 = nn.L1Loss()
    crit2 = nn.BCELoss()
    running_loss = 0.0
    for measurements,images in dataloader:
        images= images.to(device)
        optimizer.zero_grad()
        preds=model.forward(images) # 
        loss1 = crit1(preds["lane_dist"], measurements["lane_dist"].to(device))    # can i pass a     
        loss2 = crit1(preds["route_angle"], measurements["route_angle"].to(device))
        loss3 = crit1(preds["tl_dist"], measurements["tl_dist"].to(device))
        # print(preds["tl_state"].shape ,measurements["tl_state"].float().shape )
        loss4 = crit2(preds["tl_state"].float(), measurements["tl_state"].float().to(device))
        loss = loss1 + loss2 + loss3 + loss4 
        loss.backward() #calculate derivatives 
        running_loss += loss.item()
        optimizer.step()
        wandb.log({'Train/lane_dist_loss': loss1.mean().item(),
                   'Train/route_angle_loss': loss2.mean().item(),
                   'Train/tl_dist_loss': loss3.mean().item(),
                   'Train/tl_state_loss': loss4.mean().item(),
                   'Train/total_loss': loss.mean().item(),
                   "step" : global_step    
                 })
        global_step += 1
    return running_loss/len(dataloader)

def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    plt.plot(torch.tensor(train_loss).detach().cpu().numpy(),'r')
    plt.plot(torch.tensor(val_loss).detach().cpu().numpy(), 'b')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Losses-affordances')
    plt.legend(['Train loss', 'Test loss'])
    plt.savefig('Losses-affordances-20epochs.png')


def main():
    wandb.init(project='cvad-hw1-Affordances', name='my model-Affordances prediction')
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/userfiles/ssafadoust20/expert_data/train" # inside there is a measurement and rgb
    val_root = "/userfiles/ssafadoust20/expert_data/val"
    model = AffordancePredictor().to(device)
    T = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
    train_dataset = ExpertDataset(train_root, T)
    val_dataset = ExpertDataset(val_root, T)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 20
    batch_size = 256
    save_path = "pred_model-20epochs.ckpt"
    # print(train_dataset.shape)
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
