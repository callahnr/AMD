from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.hub import download_url_to_file


catalog = {
    'deep_sig_2018_train':{
        'path': 'data/dp_sig_training.hdf5',
    },
    'deep_sig_2018_test':{
        'path': 'data/dp_sig_testing.hdf5',
    }
}




class DeepSig2018(Dataset):
    ''' 
    DeepSig2018_Train Pytorch Dataset Class
    Loads data from DeepSig.ai 2018 dataset
  - f['X'] = 2.5M array of 1024 intervals of 2 I/Q datapoints
  - f['Y'] = 2.5M array of 24 modulation types one-hot-encoded
  - f['Z'] = SNR level in dB --> [['-2dB']]
 
    Assigns the modulation type as the data label
    '''  
    
    classes = [ "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", 
                "32PSK", "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", 
                "32QAM", "64QAM", "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", 
                "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]


    def __init__(   self, 
                    transform=None, 
                    download=True, 
                    split="train", 
                    progress=True, 
                    min_snr=-float('inf'),
                    max_snr=float('inf'),
                    one_hot=True
                    # mod_types=classes
                    ):
        super().__init__()
        # print(f'{os.path}')
        self.transform = transform
        
        if split=="train":
            f_path = catalog['deep_sig_2018_train']['path']
        else:
            f_path = catalog['deep_sig_2018_test']['path']

        if not os.path.isfile(f_path):
            if download:
                self.download(progress=progress)
            else:
                assert False, "No dataset has been downloaded"

        
        #  LOAD DISK DATA TO VARIABLES
        raw_data = h5py.File(f_path, "r")
        self.X = np.array(raw_data['X'])
        self.mod_type = np.array(raw_data['Y'])
        self.SNR = np.array(raw_data['Z'])
        self.SNR = self.SNR.squeeze(1)
        self.one_hot = one_hot

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.index = np.arange(len(self.X))
        self.filter_SNR(self.min_snr, self.max_snr)

    def filter_SNR(self, min_snr, max_snr):
        ''' 
        Filters the object to only contain data 
        with SNR levels between max_snr and min_snr.
        
        Args:
        - max_snr(float): Maximum level of SNR to retain
        - min_snr(float): Minimum level of SNR to retain
       
        '''
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.index, = np.where((self.SNR >= self.min_snr)&(self.SNR <= self.max_snr))
        
    @staticmethod
    def download(progress=True):
        url = catalog['deep_sig_2018_train']['url']
        path = catalog['deep_sig_2018_train']['path']
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            download_url_to_file(url, path, progress=progress)

    def __len__(self):
        return len(self.X[self.index])
    
    def __getitem__(self, idx):
        tru_idx = self.index[idx]
        # data = self.X[tru_idx]
        data = self.X[tru_idx].T
        label = self.mod_type[tru_idx]

        if not self.one_hot:
            label = label.argmax()

        if self.transform:
            data = self.transform(data)
            
        return torch.tensor(data), torch.tensor(label, dtype=torch.float)

def test():
    ds = DeepSig2018(split="train")
    x,y = ds[0]
    assert y.shape == torch.Size([24]), "Labels to default to one-hot-encoding with 24 classes."
    assert y.dtype == torch.float, "Labels should be torch.float datatype."

if __name__ == "__main__":
    test()