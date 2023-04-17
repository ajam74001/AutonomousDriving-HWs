from torch.utils.data import Dataset
import json
import os 
from skimage import io 
from PIL import Image
class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root, transform= None):
        self.data_root = data_root
        self.transform= transform
        # Your code here
        self.measurements_dir = data_root+'/measurements/'
        self.image_dir = data_root+ '/rgb/'
        self.measurements = os.listdir(self.measurements_dir)
        self.images = os.listdir(self.image_dir)

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
#         print(type(os.path.join(self.measurements_dir, self.measurements[index])))
        with open(os.path.join(self.measurements_dir, self.measurements[index]), 'r') as f:
            m = json.load(f)
        # rgb = io.imread(os.path.join(self.image_dir,self.images[index]))
        rgb = image = Image.open(os.path.join(self.image_dir,self.images[index]))
        rgb = image.convert("RGB")
        # print(rgb.shape)
        if self.transform:
            rgb = self.transform(rgb)
        return m, rgb
    def __len__(self):
        return len(self.measurements)