import torch
import torch.nn as nn
import torchvision.models as models 


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super(AffordancePredictor, self).__init__()
        # self.commands = commands
        self.resnet18 = models.resnet18(pretrained=True) ## get the last feature vector not the logits
        num_fltrs = self.resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.fc = nn.Linear(512, 4)
        self.relu = nn.ReLU()
    def forward(self, img):
        img = self.resnet18(img) #img: b x 512 x 1 x 1
        img = self.relu(img).squeeze()
        # print(img.shape)
        img = self.fc(img)
        # print(img.shape)

        return{"lane_dist": img[:,0].squeeze(),
                "route_angle": img[:,1].squeeze(), 
                "tl_dist":img[:,2].squeeze(), 
                "tl_state": torch.sigmoid(img[:,3].squeeze())}

