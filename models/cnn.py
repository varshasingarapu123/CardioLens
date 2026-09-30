import torch
from torch import nn

class ECGCNN(nn.Module):
    def __init__(self,n_classes=5):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(1,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,64,7,padding=3),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(64,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.classifier=nn.Linear(128,n_classes)
    def forward(self,x): return self.classifier(self.features(x).squeeze(-1))
