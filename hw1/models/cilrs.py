import torch
import torch.nn as nn
import torchvision.models as models 
# input: RGB image and the vehicle speed and a high level command
# output: action and the vehicle speed 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS,self).__init__()
        self.resnet18 = models.resnet18(pretrained=True) ## get the last feature vector not the logits
        num_fltrs = self.resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.speed_fc = nn.Sequential(
                nn.Linear(1, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
            )# 
        self.emb_fc = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
            ) 
        self.control_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])
        self.speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[1]),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        self.relu = nn.ReLU()
    def forward(self, img, command, speed):
        img = self.resnet18(img)
        img = self.relu(img).squeeze(-1).squeeze(-1) # b x 512
        # print(img.shape)
        speed = self.speed_fc(speed.unsqueeze(1).float()) # b x 128
        # print(speed.shape)
        emb= torch.cat([img,speed], dim =1) # b x 128+512
        # print(emb.shape)
        emb = self.emb_fc(emb) # b x 512
        
        pred_control = torch.zeros((img.shape[0],3)).to(device)
        for c in torch.unique(command):
            pred_control[command == c] = self.control_branches[c](emb[command == c ])
        # print(pred_control.shape) # b x 3
        pred_speed = self.speed_branch(img)
        # we can pass the pred_control element to relevant activation functions to make the training more stable
        # steer -> tanh() cuz the steer is between -1, 1 & throttle, brake -> sigmoid cuz brake is between 0-1
        # However as i had a stable  training, i ignored this step
        return pred_control, pred_speed
