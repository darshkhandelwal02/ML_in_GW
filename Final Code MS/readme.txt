"Quantum and Classical Machine Learning Techniques for Enhancing Gravitational Wave Science"

Author - Darsh Khandelwal, Department of Physics, IISER Bhopal. 
Supervisor - Dr. Kuntal Roy, Department of Electrical Engineering and Computer Science (EECS). 


This is the readme file to the code provided in this folder. 

Download the folder Final Code MS and start running the code then. 

Requirements:

1. qiskit verison : 1.3.2



Dataset: 
There are folders in the dataset folder located at the "Darsh" folder in the blue hard drive. 
Below is the description of each folder in the dataset and where they are used in the code. 


Folders: 
1. sxs - This folder contains the dataset of all the waveforms, preprocessed waveforms and the fft of each waveform.

    Folder Structure: It contains several folders named as
        1. preprocess_M: Containing the preprocessed waveforms of all mass ratios but same total mass (M). 
        2. fft_M: Containing the fft of the preprocessed waveforms for all mass ratios but same total mass (M). 
        3. bbh.csv : The file which contains the binary parameters of NR signals imported from SXS catalog (2476 waveforms, normalized for different M)
        4. Raw : The folder which contains the raw data downloaded from the SXS catalog of NR signals. 

2. spin - This folder contains the dataset of the saved neural network models of ML for the Spin BBH systems.

        Folder Structure: It contains few files of the model and the dataset of the model. 
            1. spin_model_final.keras - Final ML model for the Spin BBH systems
            2. spin_model_final_history.h5 - File which contains the list of the training and validation loss for each epoch. 
            3. spin_model_final_metrics.h5 - File containing MSE and R2 score for the train and test set. 
            4. spin_model_final_predictions.h5 - File containing predictions of the model 
            5. spin_model_final_target.h5 - File containing the target values of the model
            6. Spin_M40-M95.csv - CSV file containing the NN table of Spin BBH systems

3. spinless - This folder contains the dataset of the saved neural network models of ML for the Spinless BBH systems.

        Folder Structure: It contains few files of the model and the dataset of the model. 
            1. spinless_model1_m1_m2_final.keras - Final ML model 1 for the Spinless BBH systems
            2. spinless_model1_m1_m2_final_history.h5 - File which contains the list of the training and validation loss for each epoch. 
            3. spinless_model1_m1_m2_final_metrics.h5 - File containing MSE and R2 score for the train and test set. 
            4. spinless_model1_m1_m2_final_pred.h5 - File containing predictions of the model 
            5. spinless_model2_M_q_final.keras - Final ML model 2 for the Spinless BBH systems
            6. spinless_model2_M_q_final_history.h5 - File which contains the list of the training and validation loss for each epoch. 
            7. spinless_model2_M_q_final_metrics.h5 - File containing MSE and R2 score for the train and test set. 
            8. spinless_model2_M_q_final_pred.h5 - File containing predictions of the model 
            9. Zero_Spin_M40-M95.csv - File containing the Spinless BBH parameters table for NN model.

Files: 

1. ML_in_GW: Main file which contains all the code for all the models. 



