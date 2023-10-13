# Rapid extraction of respiratory waveforms from photoplethysmography: A deep corr-encoder approach
# training script
# Harry J Davies, 2023

# import dependencies 
import matplotlib
import scipy.signal
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mat73
import random
import math
import scipy.signal as sig
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# load the data
data_capno = mat73.loadmat('D:/D_copy/capnobase/segments_all_standardised.mat')
data_co2 = data_capno['segs_co2']
data_ppg = data_capno['segs_ppg']

# define number of epochs and batch size
num_epochs = 80
batch_size = 30

#define number of kernels per layer
n_in, n_out = 1, 8
n_out2 = 8
n_out3 = 8
n_outputs = 1

# define kernel lengths, padding, dilation, stride, and dropout
kernel_size = 150
kernel_size2 = 75
kernel_size3 = 50
padding = 20
dilation = 1
stride = 1
dropout_val = 0.5
padding2 = 20
padding3 = 10
dilation2 = 1
dilation3 = 1
stride2 = 1
stride3 = 1

# set a seed for evaluation (optional)
seed_val = 55
print("Seed")
print(seed_val)
torch.manual_seed(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)

# set the learning rate for Adam optimisation
learning_rate = 0.001

#define the model
class Correncoder_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_out, n_out2, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_out2, n_out3, kernel_size=kernel_size3, padding=padding3),
            nn.Sigmoid(),
            nn.Dropout(dropout_val)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(n_out3, n_out2, kernel_size=kernel_size3, padding=padding3),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(n_out2, n_out, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(n_out, n_in, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out



# a function that gives a rough indication of breaths per minute error by examining the crossings of 0.5 
# this assumes that the respiratory reference is normalised between 0 and 1.
def breaths_per_min_zc(output_array_zc, input_array_zc):
    peak_count_output = []
    peak_count_cap = []
    for ind_output in range(output_array_zc.shape[0]):
        output_array_zc_temp = output_array_zc[ind_output, 0, :]
        input_array_zc_temp = input_array_zc[ind_output, :]
        output_array_zc_temp = output_array_zc_temp - 0.5
        input_array_zc_temp = input_array_zc_temp - 0.5
        zero_crossings_output = ((output_array_zc_temp[:-1] * output_array_zc_temp[1:]) < 0).sum()
        zero_crossings_input = ((input_array_zc_temp[:-1] * input_array_zc_temp[1:]) < 0).sum()
        peak_count_output.append(zero_crossings_output)
        peak_count_cap.append(zero_crossings_input)
        # breaths_per_min_output = (zero_crossings_output / 2)*6.25
    peak_count_output = np.array(peak_count_output)
    peak_count_cap = np.array(peak_count_cap)
    #6.5 is used ot scale up to 1 minute, as each segment here is 60/6.5 seconds long.
    mean_error = ((np.mean(peak_count_output - peak_count_cap)) / 2) * 6.5 
    mean_abs_error = ((np.mean(np.abs(peak_count_output - peak_count_cap))) / 2) * 6.5
    return mean_abs_error, mean_error

# iterate through 42 subjects for training.
# the data is non shuffled and stacked with each of the 42 subjects sharing an equal proportion
# if data is not in this format and LOSO training is still needed, adapt to isolate one subject for testing and train on other subjects
kf = KFold(42)
kf.get_n_splits(data_ppg)
sub_num = 1
for train_index, test_index in kf.split(data_ppg):
    trainX, testX = data_ppg[train_index, :], data_ppg[test_index, :]
    trainy, testy = data_co2[train_index, :], data_co2[test_index, :]

    # print which subject is current test subject
    print(sub_num)

    # shuffle training data
    trainX, trainy = shuffle(trainX, trainy)

    # set a model path for saving the trained pytorch model weights
    model_path = "D:/D_copy/capnobase_code/model_sub" + str(sub_num) + ".pth"

    # initialise new model
    model = Correncoder_model()
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # ensure correct shape, can also transpose here instead of reshaping
    L_in = trainX.shape[-1]
    trainX = trainX.reshape((trainX.shape[0], 1, L_in))
    testX = testX.reshape((testX.shape[0], 1, L_in))

    total_step = trainX.shape[0]

    # transformation of data into torch tensors
    trainXT = torch.from_numpy(trainX.astype('float32'))
    # trainXT = trainXT.transpose(1,2).float() #input is (N, Cin, Lin) = Ntimesteps, Nfeatures, 128
    trainyT = torch.from_numpy(trainy.astype('float32'))
    testXT = torch.from_numpy(testX.astype('float32'))

    testyT = torch.from_numpy(testy.astype('float32'))
    # used for input to the breaths per minute calculator
    input_array = testyT.cpu().detach().numpy()

    loss_list = []
    acc_list = []
    acc_list_test_epoch = []
    test_error = []

    #begin training loop
    for epoch in range(num_epochs):
        for i in range(total_step // batch_size):  # split data into batches
            trainXT_seg = trainXT[i * batch_size:(i + 1) * batch_size, :, :]
            trainyT_seg = trainyT[i * batch_size:(i + 1) * batch_size, None]
            # Run the forward pass
            outputs = model(trainXT_seg)

            loss = criterion(outputs, trainyT_seg)

            #loss = d_loss_output[0]

            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
        # calculate test error at the end of each epoch
        test_output = model(testXT)
        loss_test = criterion(test_output, testyT[:, None])
        test_error.append((loss_test.item()))

        output_array = test_output.cpu().detach().numpy()

        mean_error_bpm = breaths_per_min_zc(output_array, input_array)

        print("Test sub")
        print(sub_num)
        print("Epoch")
        print(epoch)
        print("Training loss")
        print(loss)
        print("Test loss")
        print(loss_test)
        print("Peaks error abs")
        print(mean_error_bpm[0])
        print("Peaks error bias")
        print(mean_error_bpm[1])

    # save the PyTorch model files
    torch.save(model.state_dict(), model_path)
    sub_num = sub_num + 1




