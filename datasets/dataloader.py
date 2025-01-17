import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ..utils.utils import read_wav_np


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files
    
    
def create_dataloader(hp, args, train):
    dataset = MelFromDisk(hp, args, train)

    if train==1:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)


class MelFromDisk(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        if train==1:
            self.path = hp.data.train
        elif train==0:
            self.path = hp.data.validation
        elif train==-1:
            self.path = hp.data.total
            
        #self.wav_list = glob.glob(os.path.join(self.path, '**', '*.wav'), recursive=True)
        self.wav_list = files_to_list(self.path)
        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length + 2
        self.mapping = [i for i in range(len(self.wav_list))]

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        if self.train==1:
            idx1 = idx
            idx2 = self.mapping[idx1]
            return self.my_getitem(idx1), self.my_getitem(idx2)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        if os.environ.get('CUSTOM',''):
            #print("enter",flush=True)
            melpath = self.wav_list[idx]
            npzzz = np.load(melpath)
            audio = npzzz['audio']
            mel = npzzz['mel'].T
            
            if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
                audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                        mode='constant', constant_values=0.0)

            audio = torch.from_numpy(audio).unsqueeze(0)
            mel = torch.from_numpy(mel).squeeze(0)

            if self.train==1:
                #print("=============================",flush=True)
                
                max_mel_start = mel.size(1) - self.mel_segment_length
                #print("max_mel_start: ",max_mel_start,flush=True)
                mel_start = random.randint(0, max_mel_start)
                #print("mel_start: ",mel_start,flush=True)
                mel_end = mel_start + self.mel_segment_length
                #print("mel_end: ",mel_end,flush=True)
                mel = mel[:, mel_start:mel_end]
                #print("mel_shape: ",mel.shape,flush=True)
                audio_start = mel_start * self.hp.audio.hop_length
                #print("audio_start: ",audio_start,flush=True)
                audio = audio[:, audio_start:audio_start+self.hp.audio.segment_length]
                #print("audio_shape: ",audio.shape,flush=True)
                #print("=============================",flush=True)
            audio = audio + (1/32768) * torch.randn_like(audio)
            return mel, audio
        
        wavpath = self.wav_list[idx]
        melpath = wavpath.replace('.wav', '.mel')
        sr, audio = read_wav_np(wavpath)
        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio).unsqueeze(0)
        mel = torch.load(melpath).squeeze(0)

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hp.audio.hop_length
            audio = audio[:, audio_start:audio_start+self.hp.audio.segment_length]

        audio = audio + (1/32768) * torch.randn_like(audio)
        return mel, audio
