##############################################################
file_name = "lstm_ae"
look_back = 64
batch_size = 128
epochs = 40
learning_rate = 0.0001
#######################################################################

from gwpy.timeseries import TimeSeries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import sklearn as sk
import h5py
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import gwpy
from gwpy.signal.filter_design import bandpass
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from gwpy.plot import Plot
import scipy
from scipy import signal
from scipy.signal import decimate
import sxs
import os
import pickle
import sys

script_path = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the script
data_dir = script_path + f"/dataset/{file_name}/"
main_dir = script_path + f"/data/"


# Define parameters for the Gaussian white noise
mean = 0
std_dev = 1
duration = 64  # duration in seconds
fs = 4096  # sampling frequency in Hz
num_samples = duration * fs  # total number of samples

# Generate Gaussian white noise
gauss_noise = np.random.normal(mean, std_dev, num_samples)

# Create a time series object
noise = TimeSeries(gauss_noise, sample_rate=fs)
noise_norm = noise/np.max(np.abs(noise))
t_noise = np.linspace(0, duration, num_samples)

############## Saving the Dataset ##################
with open(data_dir + "noise_data.pkl", "wb") as f:
    pickle.dump({"noise": noise, "noise_norm": noise_norm}, f)

# # Obtaining the Dataset from where the names of the waveforms are available 
# simulations = sxs.load("simulations")  # Load the simulations catalog
# df = simulations.dataframe  # Get the catalog as a DataFrame

# df_working = df[df["deprecated"] == False]  # Get only the non-deprecated BHBH simulations
# df_BBH = df_working[df_working["object_types"] == "BHBH"]  # Filter for binary black hole (BBH) mergers

# index = df_BBH.index.to_numpy()  # Get the list of simulation names
# new_index = [name.replace(":", "_") for name in index]  # Format names properly

with open(main_dir + "new_index.pkl", "rb") as f:
    new_index = pickle.load(f)

len_noise = len(t_noise)  # Define total noise length
waveforms = []
names = [] # Names of the waveforms which will be used in the injection
remaining_length = len_noise  # Track remaining space for waveforms

def nearest_power_of_2(n):  
    return 2 ** int(np.ceil(np.log2(n)))  # Compute nearest power of 2

i = 0
while remaining_length > 1024 and i < len(new_index):  
    with h5py.File(main_dir + 'preprocess_60/' + f'{new_index[i]}_pp.h5', 'r') as f:
        time = f['time'][:]
        wf = f['plus'][:]  # Load waveform data
        
    if len(wf) < 1024:  # Skip waveforms smaller than 1024 samples
        i += 1
        continue  
    wf = wf / np.max(np.abs(wf))  # Normalize waveform
    wf_ds = scipy.signal.decimate(wf, 2)  # Downsample by factor of 2

    if len(wf_ds) < 1024:  # Skip again if downsampled waveform is too small
        i += 1
        continue  
    new_len = nearest_power_of_2(len(wf_ds))  # Find nearest power of 2
    if new_len > remaining_length:  
        i += 1
        continue  # Skip if waveform is too long

    pad_width = (new_len - len(wf_ds)) // 2  # Compute padding width
    wf_ds = np.pad(wf_ds, (pad_width, new_len - len(wf_ds) - pad_width), mode='constant')  # Pad to nearest power of 2

    names.append(new_index[i])  # Store waveform name
    waveforms.append(wf_ds)  # Append padded waveform

    remaining_length -= len(wf_ds)  # Update remaining length
    i += 1

if remaining_length > 0:  
    waveforms.append(np.zeros(remaining_length))  # Pad with zeros if space is left
app_wf = np.concatenate(waveforms)  # Concatenate all waveforms
app_wf = app_wf[:len_noise]  # Ensure final waveform matches len_noise



############ Saving the Data ####################
with open(data_dir + "inj_waveforms.pkl", "wb") as f:
    pickle.dump({"waveforms": waveforms, "names_waveforms": names, "time": t_noise,
                  "inj_data": app_wf}, f)


signal = noise_norm + app_wf
# signal = noise + app_wf
signal_norm = signal / np.max(np.abs(signal))
app_wf_norm = app_wf / np.max(np.abs(signal))
norm_noise = noise_norm / np.max(np.abs(signal))

############ Saving the Data ####################
with open(data_dir + "signal.pkl", "wb") as f:
    pickle.dump({"signal": signal, "signal_norm": signal_norm}, f)


def create_seq(data, look_back=64, step=1):
    sequences = []
    for i in range(0, len(data) - look_back + 1, step):
        sequences.append(data[i:i+look_back])
    return np.array(sequences).reshape(-1, look_back, 1)

def reverse_seq(seq):
    recon_signal = list(seq[0, :, 0])  # Start with the first sequence
    for i in range(1, seq.shape[0]):
        recon_signal.append(seq[i, -1, 0])  # Append only the last element of each subsequent sequence
    return np.array(recon_signal)


# Create overlapping sequences (sliding window) with look_back=64 and step=1
# look_back = 64
X = create_seq(signal_norm, look_back=look_back)
y = create_seq(app_wf_norm, look_back=look_back)

# print("Shape of X (simulated noise with injected waveforms):", X.shape)
# print("Shape of y (clean target waveform):", y.shape)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = False)

# print("Length of X_train data:", X_train.shape)
# print("Length of X_test data:", X_test.shape)
# print("Length of y_train data:", y_train.shape)
# print("Length of y_test data:", y_test.shape)

################## Save the Dataset ###############
with open(data_dir + "train_test_dataset.pkl", "wb") as f:
    pickle.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}, f)

# Build the LSTM autoencoder model

# batch_size = 128
# epochs = 30

model = Sequential()
model.add(Input(shape=(look_back, 1)))
# model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(LSTM(32, activation='tanh', return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(16, activation='tanh', return_sequences= True))
model.add(LSTM(8, activation = 'tanh', return_sequences = False))
model.add(RepeatVector(look_back))
model.add(LSTM(8, activation = 'tanh', return_sequences = True))
model.add(LSTM(16, activation='tanh', return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(32, activation='tanh', return_sequences=True))
# model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(1)))  # Output shape same as input
# Define learning rate schedule


# Configure logging
logging.basicConfig(filename= data_dir + f"training_log_{file_name}.txt", 
                    level=logging.INFO, format="%(asctime)s - %(message)s")

class LossLoggerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Epoch {epoch+1}: Loss={logs['loss']}, Val_Loss={logs.get('val_loss')}")

# Use the callback
loss_logger = LossLoggerCallback()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate= learning_rate,
    decay_steps = 10000,
    decay_rate=0.96,
    staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='mse')
model.summary()
history = model.fit(X_train, y_train, epochs= epochs, batch_size=batch_size, 
                    validation_data=(X_test, y_test), callbacks=[loss_logger], verbose=1)

################### Save the model ################
model.save(data_dir + f"model_{file_name}.keras")
################### Save the history ################
with open(data_dir + f"hist_model_{file_name}.pkl", "wb") as f:
    pickle.dump(history.history, f)


# Predict on test data
predicted_waveform = model.predict(X_test)

# Test Dataset
test_sig = reverse_seq(y_test)
test_pred = reverse_seq(predicted_waveform)
signal_sig = reverse_seq(X_test)

# Train Dataset
train_sig = reverse_seq(y_train)
train_pred = reverse_seq(model.predict(X_train))
signal_sig_train = reverse_seq(X_train)

# Time vector
t_train = np.linspace(0, len(train_sig)/fs, len(train_sig))
t_test = np.linspace(0, len(test_sig)/fs, len(test_sig))

# Mean Squared Error
mse_train = mean_squared_error(train_sig, train_pred)
mse_test = mean_squared_error(test_sig, test_pred)

# Normalized Cross-Correlation
def normalized_cross_correlation(x_true, x_pred):
    """Compute Normalized Cross-Correlation (NCC) between true and predicted waveforms."""
    x_true = x_true.flatten()
    x_pred = x_pred.flatten()

    numerator = np.sum(x_true * x_pred)
    denominator = np.sqrt(np.sum(x_true**2) * np.sum(x_pred**2))

    return numerator / denominator if denominator != 0 else 0

# Compute NCC
ncc = normalized_cross_correlation(test_sig, test_pred)

# Save the dataset
dataset = {
    "test_pred": test_pred,
    "train_pred": train_pred,
    "test_sig": test_sig,
    "train_sig": train_sig
    }

# Save the dataset as a pickle file
with open(data_dir + f"model_{file_name}_results.pkl", "wb") as f:
    pickle.dump(dataset, f)

# Create the log file
with open(data_dir + f"model_{file_name}_log.txt", "w") as log_file:
    # Write dataset information
    log_file.write("### Dataset Information ###\n")
    log_file.write(f"Number of waveforms used: {len(waveforms)}\n")
    log_file.write(f"Waveforms used: {names}\n")
    log_file.write(f"Train dataset length: {len(train_sig)}\n")
    log_file.write(f"Test dataset length: {len(test_sig)}\n")
    log_file.write("\n")

    # Write model summary
    log_file.write("### Model Summary ###\n")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    log_file.write("\n".join(model_summary))
    log_file.write("\n")

    # Write prediction accuracy
    log_file.write("### Prediction Accuracy ###\n")
    log_file.write(f"Mean Squared Error (Train): {mse_train}\n")
    log_file.write(f"Mean Squared Error (Test): {mse_test}\n")
    log_file.write(f"Normalized Cross-Correlation (NCC): {ncc}\n")
    log_file.write("\n")

    # Write additional information
    log_file.write("### Additional Information ###\n")
    log_file.write(f"Standard Deviation of Noise: {std_dev}\n")
    log_file.write("\n")











