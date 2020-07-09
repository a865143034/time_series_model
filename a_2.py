#coding:utf-8
import jieba.posseg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(310,128),
            nn.ReLU(),
            nn.Linear(128,128))
        self.class_classifier = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,3),
            nn.ReLU())
        self.domain_classifier = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,2),)

        def forward(self, input_data):
            # input_data = input_data.view(31,10)#
            feature = self.feature(input_data)
            # feature = feature.view(-1, 50 * 4 * 4)#
            #reverse_feature = ReverseLayerF.apply(feature, 0.1)  #
            class_output = self.class_classifier(feature)
            #domain_output = self.domain_classifier(reverse_feature)

            return class_output#, domain_output