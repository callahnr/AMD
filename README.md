# Curriculum based transfer learning for Automatic Modulation Detection
ML/DL Automatic Modulation Detection Using Pytorch

Running AMD_Trainer.py will download the data set from https://www.kaggle.com/pinxau1000/radioml2018?select=datasets.desktop provided you have a kaggle API key. If not, manually download the dataset and place the extracted files in the ./data/ folder and the trainer will split the master .hdf5 file.

The current default configuration is set with a Binary Cross Entrophy Loss function for experimentation. Prior results with higher accuracy were collected using Cross Entrophy Loss.
