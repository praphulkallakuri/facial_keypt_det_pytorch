#! /usr/bin/env python

import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torchsummary import summary
import numpy as np
import pandas as pd
import os,glob,cv2
from torch.utils.data import TensorDataset, DataLoader, Dataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn import cluster
from copy import deepcopy
from sklearn.model_selection import train_test_split


device = 'cuda'
root_dir = '/home/praphul/pytorch/vgg16/P1_Facial_Keypoints/data/training/'
img_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
data = pd.read_csv('/home/praphul/pytorch/vgg16/P1_Facial_Keypoints/data/training_frames_keypoints.csv')
# print(data.iloc[5,1:])

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='*', c='r')
    plt.pause(0.001)

class Faces(Dataset):
    def __init__(self, df):
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = root_dir + self.df.iloc[index,0]
        # print(img_path)
        img = cv2.imread(img_path)/255
        kp = deepcopy(self.df.iloc[index,1:].tolist())
        # print(kp)
        kp_x = (np.array(kp[0::2])/img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2])/img.shape[1]).tolist()

        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2)
        # print(kp2.shape)
        img = self.preprocess_input(img)
        return img, kp2

    def preprocess_input(self,img):
        img = cv2.resize(img, (224,224))
        img = torch.tensor(img).permute(2,0,1)
        img = self.normalize(img).float()
        return img.to(device)

    def load_image(self,index):
        img_path = root_dir + self.df.iloc[index,0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img = cv2.resize(img, (224,224))
        return img

train, test = train_test_split(data, test_size=0.2, random_state=101)
train_dataset = Faces(train.reset_index(drop=True))
test_dataset = Faces(test.reset_index(drop=True))

train_loader = DataLoader(train_dataset,batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


def get_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.avgpool = nn.Sequential(nn.Conv2d(512,512,3),
                                  nn.MaxPool2d(2),
                                  nn.Flatten())
    model.classifier = nn.Sequential(
                                        nn.Linear(2048,512),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(512, 136),
                                        nn.Sigmoid()
                                    )
    criterion = nn.L1Loss() #mean absolute error
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    return model.to(device), criterion, optimizer          


model, criterion, optimizer = get_model()

def train_bacth(img, kps, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    loss.backward()
    optimizer.step()
    return loss

def validate_data(img, kps, model, criterion):
    model.eval()
    with torch.no_grad():
        _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    return _kps, loss

train_loss, test_loss = [],[]
epochs = 1
for epoch in range(epochs):
    print(f"epoch {epoch+1} : 30")
    epoch_train_loss, epoch_test_loss = 0,0
    for index, (img,kps) in enumerate(train_loader):
        loss = train_bacth(img, kps, model, optimizer, criterion)
        epoch_train_loss += loss.item()
    epoch_train_loss /= (index+1)
    print('epoch train loss: ', epoch_train_loss)

    for index, (img,kps) in enumerate(test_loader):
        ps,loss = validate_data(img, kps, model, criterion)   
        epoch_test_loss +=  loss.item()
    epoch_test_loss /= (index+1)    
    train_loss.append(epoch_train_loss)
    test_loss.append(epoch_test_loss)    

epochs = np.arange(30)+1

plt.plot(epochs, train_loss, 'bo', label='training_loss')
plt.plot(epochs,test_loss, 'r', label='test_loss')
plt.title('training and test loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid('off')
plt.show()
