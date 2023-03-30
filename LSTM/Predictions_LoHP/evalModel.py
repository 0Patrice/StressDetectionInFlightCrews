from __future__ import print_function
import argparse
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import mean, sqrt, square, arange
import pandas as pd

import logging
import csv

import os
from os.path import exists
import datetime
import time
import shutil

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='evalLog.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


class Sequence(nn.Module):

    def __init__(self):
        super(Sequence, self).__init__()
        input_size = 1
        self.input_size = input_size

        hidden_layers1 = 256
        hidden_layers2 = 128
        hidden_layers3 = 64
        hidden_layers4 = 32
        hidden_layers5 = 16

        self.hidden_layers1 = hidden_layers1
        self.hidden_layers2 = hidden_layers2
        self.hidden_layers3 = hidden_layers3
        self.hidden_layers4 = hidden_layers4
        self.hidden_layers5 = hidden_layers5

        self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_layers1).cuda()
        self.lstm2 = nn.LSTMCell(self.hidden_layers1, self.hidden_layers2).cuda()
        self.lstm3 = nn.LSTMCell(self.hidden_layers2, self.hidden_layers3).cuda()
        self.lstm4 = nn.LSTMCell(self.hidden_layers3, self.hidden_layers4).cuda()
        self.lstm5 = nn.LSTMCell(self.hidden_layers4, self.hidden_layers5).cuda()

        self.linear = nn.Linear(self.hidden_layers5, 1).cuda()

    def forward(self, inputData):
        outputs = []
        h_t = torch.zeros(inputData.size(0), self.hidden_layers1, dtype=torch.double).cuda()
        c_t = torch.zeros(inputData.size(0), self.hidden_layers1, dtype=torch.double).cuda()

        h_t2 = torch.zeros(inputData.size(0), self.hidden_layers2, dtype=torch.double).cuda()
        c_t2 = torch.zeros(inputData.size(0), self.hidden_layers2, dtype=torch.double).cuda()

        h_t3 = torch.zeros(inputData.size(0), self.hidden_layers3, dtype=torch.double).cuda()
        c_t3 = torch.zeros(inputData.size(0), self.hidden_layers3, dtype=torch.double).cuda()

        h_t4 = torch.zeros(inputData.size(0), self.hidden_layers4, dtype=torch.double).cuda()
        c_t4 = torch.zeros(inputData.size(0), self.hidden_layers4, dtype=torch.double).cuda()

        h_t5 = torch.zeros(inputData.size(0), self.hidden_layers5, dtype=torch.double).cuda()
        c_t5 = torch.zeros(inputData.size(0), self.hidden_layers5, dtype=torch.double).cuda()

        for input_t in inputData.split(1, dim=1):

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))
            h_t5, c_t5 = self.lstm5(h_t4, (h_t5, c_t5))

            output = self.linear(h_t5)
            outputs += [output]

        return torch.cat(outputs, dim=1)

if __name__ == '__main__':

    columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp', 'UTC', 'DA', 'Conf']
    columns_ruhe = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp']
    # ECG Signal Steps with a range (0, 246)
    ecg_norm = 1 / 246

    # Stress Level Track: Scale from 0-10 to 0-1
    slt_norm = 1 / 10

    model_state_path = 'SPECGmodelNEO.pt'

    seq = Sequence()
    seq.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    seq.eval()
    seq.double()

    train_data_path = "/home/lori_pa/ml/ValidationDataGMLF_MRFmV/"

    # Get CPU or GPU device for Training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f'Setting Device to: {torch.cuda.get_device_name()}')
    print(f"Using {device} device")

    #valdata = pd.read_csv(train_data_example_path, delimiter=',', usecols=columns, dtype=np.double)

    for root, subdirectories, files in os.walk(train_data_path, topdown=True):
        for file in files:
            logging.info(f'Validating Model with {file}')

            file_path = os.path.join(root, file)
            if file.endswith('RuheEKG.csv'):
                print('Ruhe EKG')

                # data = np.genfromtxt(file_path, delimiter=";", skip_header=1)
                valdata = pd.read_csv(file_path, delimiter=',', usecols=columns_ruhe, dtype=np.double)

                print(valdata.shape)
                logging.info(f'Validation Data Shape: {valdata.shape}')

                test_input = torch.tensor([valdata['II'] * ecg_norm]).to(device)

                test_target = torch.ones((1, len(valdata['II'])), dtype=torch.double).to(device)
                test_target = test_target * 0.05
                # train_target

                # print(train_target)

            else:
                # data = np.genfromtxt(file_path, delimiter=";", skip_header=1)
                valdata = pd.read_csv(file_path, delimiter=',', usecols=columns, dtype=np.double)

                print(valdata.shape)
                logging.info(f'Validation Data Shape: {valdata.shape}')

                test_input = torch.tensor([valdata['II'] * ecg_norm]).to(device)
                test_target = torch.tensor([valdata['DA'] * slt_norm]).to(device)

            print(test_input.shape)
            print(test_target.shape)
            logging.info(f'Test Input Shape: {test_input.shape}')
            logging.info(f'Test Target Shape: {test_target.shape}')

            with torch.no_grad():
                pred = seq(test_input)

                predTensor = pred.cpu()

                logging.info(pred.shape)
                logging.info(predTensor.shape)
                
                prediction_df = pd.DataFrame(predTensor.numpy())
                prediction_df.to_csv(f'prediction_{file}')

