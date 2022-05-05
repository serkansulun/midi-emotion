import numpy as np
import torch
from tqdm import tqdm
from data.data_processing import  tensor_to_ind_tensor
import sys
sys.path.append("..")

import os

"""
Loads ALL data for exhaustive evaluation
"""

class LoaderExhaustive:

    def __init__(self, data_folder, data, input_len, conditioning, save_input_dir=None, pad=True,
                use_start_token=True, use_end_token=False, always_use_discrete_condition=False,
                debug=False, overfit=False, regression=False,
                max_samples=None, use_cls_token=True):

        self.data_folder = data_folder
        self.save_input_dir = save_input_dir
        self.input_len = input_len
        self.overfit = overfit
        self.one_sample = None
        self.conditioning = conditioning
        self.regression = regression
        

        if debug or overfit:
            data_folder = data_folder + "_debug"

        self.data = data

        maps_file = os.path.join(data_folder, "maps.pt")
        self.maps = torch.load(maps_file)

        self.pad_token = '<PAD>' if pad else None
        self.start_token = '<START>' if use_start_token else None
        self.end_token = '<END>' if use_end_token else None
        self.cls_token = "<CLS>"


        extra_tokens = []
        if self.conditioning == "continuous_token":
            # two condition tokens will be concatenated later
            self.input_len -= 2
        elif self.conditioning == "discrete_token":
            # two condition tokens will be concatenated later
            self.input_len -= 2
            # add emotion tokens to mappings
            for sample in self.data:
                for label in ["valence", "arousal"]:
                    token = sample[label]
                    if token not in extra_tokens:
                        extra_tokens.append(token)
            extra_tokens = sorted(extra_tokens)
            
        if self.regression and use_cls_token:
            extra_tokens.append(self.cls_token)
            self.input_len -= 1   # cls token

        if self.regression:
            chunk_len = self.input_len
        else:
            # +1 for target
            chunk_len = self.input_len + 1

        if extra_tokens != []:
            # add to maps
            maps_list = list(self.maps["idx2tuple"].values())
            maps_list += extra_tokens
            self.maps["idx2tuple"] = {i: val for i, val in enumerate(maps_list)}
            self.maps["tuple2idx"] = {val: i for i, val in enumerate(maps_list)}

        if max_samples is not None and not debug and not overfit:
            self.data = self.data[:max_samples]

        # Chunk entire data
        chunked_data = []
        print('Constructing data loader...')
        for i in tqdm(range(len(self.data))):

            data_path = os.path.join(data_folder, "lpd_5_full_transposable", self.data[i]["file"] + ".pt")
            item = torch.load(data_path)
            song = item["bars"]

            if self.conditioning != 'none' or self.regression:
                valence = self.data[i]["valence"]
                arousal = self.data[i]["arousal"]

            if self.conditioning in ("continuous_token", "continuous_concat") or self.regression:
                condition = torch.FloatTensor([valence, arousal])
            else:
                condition = torch.FloatTensor([np.nan, np.nan])

            song = torch.cat(song, 0)
            song = tensor_to_ind_tensor(song, self.maps["tuple2idx"])
            if self.start_token is not None:
                # add start token
                start_idx = torch.ShortTensor(
                    [self.maps["tuple2idx"][self.start_token]])
                song = torch.cat((start_idx, song), 0)

            if self.conditioning == "discrete_token":
                condition_tokens = torch.ShortTensor([
                    self.maps["tuple2idx"][valence],
                    self.maps["tuple2idx"][arousal]])
                if not always_use_discrete_condition:
                    song = torch.cat((condition_tokens, song), 0)

            # split song into chunks
            song = list(torch.split(song, chunk_len))  # +1 for target
            if song[-1].size(0) != chunk_len:
                song.pop(-1)

            if self.regression and use_cls_token:
                # prepend <CLS> token
                cls_idx = torch.ShortTensor(
                    [self.maps["tuple2idx"][self.cls_token]])

                song = [torch.cat((cls_idx, x), 0) for x in song]

            if self.conditioning == "discrete_token" and always_use_discrete_condition:
                song = [torch.cat((condition_tokens, x), 0) for x in song]
                
            song = [(x, condition) for x in song]
           
            chunked_data += song

        self.data = chunked_data
        print('Data loader constructed.')

    def get_vocab_len(self):
        return len(self.maps["tuple2idx"])

    def get_maps(self):
        return self.maps

    def get_pad_idx(self):
        return self.maps["tuple2idx"][self.pad_token]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk, condition = self.data[idx]
        chunk = chunk.long()

        if self.regression:
            input_ = chunk
            target = None   # will use condition as target
        else:
            input_ = chunk[:-1]
            target = chunk[1:]

            if self.conditioning == "continuous_token":
                # pad target from left, because input will get conditions concatenated
                # their sizes should match
                target = torch.nn.functional.pad(target, (condition.size(0), 0), value=self.get_pad_idx()) 
        
        return input_, condition, target


    

        

        




