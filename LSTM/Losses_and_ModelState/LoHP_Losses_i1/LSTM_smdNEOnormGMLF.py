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
fhandler = logging.FileHandler(filename='traininglog.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[44m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[41m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


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

    def closure():
        optimizer.zero_grad()
        out = seq(train_input)
        loss = criterion(out, train_target)
        global k
        k = k + 1
        print(f'{bcolors.WARNING}iter: {k} ; loss: {loss.item()}{bcolors.ENDC}\n')
        logging.warning(f'iter: {k} ; loss: {loss.item()}')

        ts = time.time()
        fields = [datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), loss.item(), file]

        with open(rf'lossNEO_i{i + i_offset}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

        loss.backward()

        return loss


    i_offset = 0

    logging.info('Starting Main')
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=20, help='steps to run')

    opt, unknown = parser.parse_known_args()  #

    # set random seed to 0
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
        logging.info(f'No Existing Model found; Creating new one')

        seq = Sequence().to(device)

    seq.double()

    # use Mean Squared Error
    criterion = nn.MSELoss()

    #optimizer = optim.Adam(seq.parameters(), lr=0.001)
    optimizer = optim.Adam(seq.parameters(), lr=0.00000001, weight_decay=0.0000000001)

    logging.info(seq.parameters())
    logging.info(optimizer)

    # load data and make training set
    data_folder = "/home/lori_pa/ml/GMLF_MRFmV"
    learned_folder = "/home/lori_pa/ml/0_GMLF_learned"

    # ECG Signal Steps with a range (0, 246)
    ecg_norm = 1 / 246

    # Stress Level Track: Scale from 0-10 to 0-1
    slt_norm = 1 / 10

    columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp', 'UTC', 'DA', 'Conf']
    columns_ruhe = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp']

    for i in range(opt.steps):

        print(f'{bcolors.OKBLUE}STEP: {i + i_offset}{bcolors.ENDC}\n')
        logging.info(f'STEP: {i + i_offset}')

        for root, subdirectories, files in os.walk(data_folder, topdown=True):
            for file in files:
                print(file)
                logging.info(f'>> New File Started: {file}')

                # logging.info(f'Emptpy CUDA Cache')
                # torch.cuda.empty_cache()

                file_path = os.path.join(root, file)

                if file.endswith('RuheEKG.csv'):
                    print('Ruhe EKG')

                    # data = np.genfromtxt(file_path, delimiter=";", skip_header=1)
                    data = pd.read_csv(file_path, delimiter=',', usecols=columns_ruhe, dtype=np.double)

                    print(data.shape)
                    logging.info(f'Input Data Shape: {data.shape}')

                    train_input = torch.tensor([data['II'] * ecg_norm]).to(device)

                    train_target = torch.ones((1, len(data['II'])), dtype=torch.double).to(device)
                    train_target = train_target * 0.05
                    #train_target

                    #print(train_target)

                else:
                    # data = np.genfromtxt(file_path, delimiter=";", skip_header=1)
                    data = pd.read_csv(file_path, delimiter=',', usecols=columns, dtype=np.double)

                    print(data.shape)
                    logging.info(f'Input Data Shape: {data.shape}')

                    train_input = torch.tensor([data['II'] * ecg_norm]).to(device)
                    train_target = torch.tensor([data['DA'] * slt_norm]).to(device)

                print(train_input.shape)
                print(train_target.shape)

                logging.info(f'Training Input Shape: {train_input.shape}')
                logging.info(f'Training Target Shape: {train_target.shape}')

                # begin to train
                print('Let the Fun begin')
                logging.info(f'Starting Model training')

                global k
                k = 0

                optimizer.step(closure)

                print(f'{bcolors.OKGREEN}Model Learning on file {file} Compleated{bcolors.ENDC}')
                logging.info(f'Model Learning on file >>{file}<< Compleated')

                logging.info(f'saving Model for Step')
                torch.save(seq.state_dict(), 'SPECGmodelNEO.pt')

                cdDir = os.path.split(root)
                newDir = cdDir[0] + "/0_GMLF_learned/" + file
                shutil.move(file_path, newDir)
                logging.info(f'Moved {file} to {newDir}')

                torch.cuda.empty_cache()
                logging.info(f'<< End')

        logging.info(f'Compleated all files for Step {i + i_offset}')

        logging.info(f'Moving all Files from {learned_folder} to {data_folder}')
        file_names = os.listdir(learned_folder)
        for file_name in file_names:
            shutil.move(os.path.join(learned_folder, file_name), data_folder)

        logging.info('Done.')
        torch.cuda.empty_cache()

        logging.info('Let the Validation begin')

        valDataFolder = "/home/lori_pa/ml/ValidationDataGMLF_MRFmV/"
        y = []
        total_loss = []

        for root, subdirectories, files in os.walk(valDataFolder, topdown=True):
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
                    #train_target

                    #print(train_target)

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
                    # use all pred samples, but only go to 999
                    loss = criterion(pred, test_target)
                    print('test loss:', loss.item())
                    logging.info(f'Test Loss: {loss.item()}')

                    ts = time.time()
                    fields = [datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), loss.item(), file]

                    with open(rf'lossValNEO_i{i + i_offset}.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(fields)

                    predTensor = pred.cpu()

                    y = np.append(y, predTensor.detach().numpy())

                    total_loss = np.append(total_loss, loss.item())

        logging.info(f'Accumulated Validation Loss {np.sum(total_loss)}')
        logging.info('Validation Step done.')

    logging.info(f'Model Training Compleated')

    torch.save(seq.state_dict(), 'SPECGmodelNEO.pt')
    logging.info(f'Model Saved')

    logging.info(f'Good Bye! :)')
