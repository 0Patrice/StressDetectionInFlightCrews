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

import os
from os.path import exists
import datetime
import time
import csv

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='validationlog.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


class Sequence(nn.Module):

    def __init__(self):
        super(Sequence, self).__init__()
        input_size = 1
        self.input_size = input_size

        hidden_layers1 = 500
        hidden_layers2 = 400
        hidden_layers3 = 150
        hidden_layers4 = 100
        hidden_layers5 = 20

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

    def forward(self, input):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_layers1, dtype=torch.double).cuda()
        c_t = torch.zeros(input.size(0), self.hidden_layers1, dtype=torch.double).cuda()

        h_t2 = torch.zeros(input.size(0), self.hidden_layers2, dtype=torch.double).cuda()
        c_t2 = torch.zeros(input.size(0), self.hidden_layers2, dtype=torch.double).cuda()

        h_t3 = torch.zeros(input.size(0), self.hidden_layers3, dtype=torch.double).cuda()
        c_t3 = torch.zeros(input.size(0), self.hidden_layers3, dtype=torch.double).cuda()

        h_t4 = torch.zeros(input.size(0), self.hidden_layers4, dtype=torch.double).cuda()
        c_t4 = torch.zeros(input.size(0), self.hidden_layers4, dtype=torch.double).cuda()

        h_t5 = torch.zeros(input.size(0), self.hidden_layers5, dtype=torch.double).cuda()
        c_t5 = torch.zeros(input.size(0), self.hidden_layers5, dtype=torch.double).cuda()

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))
            h_t5, c_t5 = self.lstm5(h_t4, (h_t5, c_t5))

            output = self.linear(h_t5)
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs

if __name__ == '__main__':
    logging.info('Starting Validation Script')
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark=True

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1, help='steps to run')

    opt, unknown = parser.parse_known_args()#

    # set random seedto 0
    np.random.seed(0)
    torch.manual_seed(0)

    # Get CPU or GPU device for Training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f'Setting Device to: {torch.cuda.get_device_name()}')
    print(f"Using {device} device")

    # build the model
    model_state_path = 'SPECGmodelNEO.pt'

    if exists(model_state_path):
        print(f'::: Loading existing Model :::')
        logging.info(f'::: Loading existing Model :::')

        seq = Sequence().to(device)
        seq.load_state_dict(torch.load(model_state_path))
        seq.eval()

    else:
        print(f'No Existing Model found \n Creating new one: \n \n')
        logging.critical(f'No Existing Model found; ABORTING')
        exit()


    seq.double()
    criterion = nn.CrossEntropyLoss()

    # use LBFGS as optimizer since we can load the whole data to train

    #optimizer = optim.LBFGS(seq.parameters(), lr=0.08, max_iter=30, max_eval=50)
    #logging.info(optimizer)


    # begin to Validate
    print('Let the Validation begin')
    logging.info('Let the Validation begin')

    valDataFolder = "/home/lori_pa/ml/ValidationDatatMRFmV/"
    y = []
    total_loss = []

    for root, subdirectories, files in os.walk(valDataFolder, topdown=True):
        for file in files:
            logging.info(f'Validating Model with {file}')

            file_path = os.path.join(root, file)

            valdata = np.genfromtxt(file_path, delimiter=";", skip_header=1)

            print(valdata.shape)
            logging.info(f'Validation Data Shape: {valdata.shape}')

            test_input = torch.tensor([valdata[:, 1]]).to(device)
            test_target = torch.tensor([valdata[:, 2]]).to(device)

            print(test_input.shape)
            print(test_target.shape)
            logging.info(f'Test Input Shape: {test_input.shape}')
            logging.info(f'Test Target Shape: {test_target.shape}')

            with torch.no_grad():
                pred = seq(test_input)
                # use all pred samples, but only go to 999
                loss = criterion(pred, test_target)
                print('test loss:', loss.item())
                logging.info(f'Test Loss: {loss.item()}')
                
                ts = time.time()
                fields = [datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), loss.item(), file]
                
                with open(r'lossVal.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                
                predTensor = pred.cpu()
                
                y = np.append(y, predTensor.detach().numpy())
                
                total_loss = np.append(total_loss, loss.item())

    logging.info(f'Accumulated Validation Loss {np.sum(total_loss)}')
    logging.info('Validation Step done.')
    logging.info(f'Good Bye! :)')