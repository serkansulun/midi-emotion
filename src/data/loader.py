import numpy as np
import random
import torch
from data.data_processing import transpose, tensor_to_ind_tensor
from data.data_processing_reverse import tuples_to_str
import sys
sys.path.append("..")
from utils import get_n_instruments
import os

"""
Main data loader
"""

class Loader:

    def __init__(self, data_folder, data, input_len, conditioning, save_input_dir=None, pad=True,
                use_start_token=True, use_end_token=False, max_transpose=3, n_try=5,
                bar_start_prob=0.5, debug=False, overfit=False, regression=False,
                max_samples=None, min_n_instruments=3, use_cls_token=True,
                always_use_discrete_condition=False):

        self.data_folder = data_folder
        self.bar_start_prob = bar_start_prob
        self.save_input_dir = save_input_dir
        self.input_len = input_len
        self.n_try = n_try  # max number of trials to find suitable sample
        self.min_n_instruments = min_n_instruments
        self.overfit = overfit
        self.one_sample = None
        self.transpose_options = list(range(-max_transpose, max_transpose + 1))
        self.conditioning = conditioning
        self.regression = regression
        self.use_cls_token = use_cls_token
        self.pad = pad
        self.always_use_discrete_condition = always_use_discrete_condition

        self.pad_token = '<PAD>' if pad else None
        self.start_token = '<START>' if use_start_token else None
        self.end_token = '<END>' if use_end_token else None
        self.cls_token = "<CLS>"

        if debug or overfit:
            data_folder = data_folder + "_debug"

        self.data = data

        maps_file = os.path.join(os.path.abspath(data_folder + "/.."), "maps.pt")
        self.maps = torch.load(maps_file)

        extra_tokens = []
        if self.conditioning == "continuous_token":
            # two condition tokens will be concatenated later
            self.input_len -= 2
        elif self.conditioning == "discrete_token":
            # add emotion tokens to mappings
            for sample in self.data:
                for label in ["valence", "arousal"]:
                    token = sample[label]
                    if token not in extra_tokens:
                        extra_tokens.append(token)
            extra_tokens = sorted(extra_tokens)

        if self.regression and self.use_cls_token:
            extra_tokens.append(self.cls_token)

        if extra_tokens != []:
            # add to maps
            maps_list = list(self.maps["idx2tuple"].values())
            maps_list += extra_tokens
            self.maps["idx2tuple"] = {i: val for i, val in enumerate(maps_list)}
            self.maps["tuple2idx"] = {val: i for i, val in enumerate(maps_list)}
        
        if max_samples is not None and not debug and not overfit:
            self.data = self.data[:max_samples]

        # roughly / 256, but *4 for flexibility. it is later cut anyway
        self.n_bars = max(round(input_len / 256 * 4), 1)    


    def get_vocab_len(self):
        return len(self.maps["tuple2idx"])

    def get_maps(self):
        return self.maps

    def get_pad_idx(self):
        return self.maps["tuple2idx"][self.pad_token]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if not self.overfit or self.one_sample is None:
            data_path = os.path.join(self.data_folder, self.data[idx]["file"] + ".pt")
            item = torch.load(data_path)
            all_bars = item["bars"]

            n_instruments = 0
            j = 0
            while j < self.n_try and n_instruments < self.min_n_instruments:
                # make sure to have n many instruments
                # choose random bar
                max_bar_start_idx = max(0, len(all_bars) - self.n_bars - 1)
                bar_start_idx = random.randint(0, max_bar_start_idx)
                bar_end_idx = min(len(all_bars), bar_start_idx + self.n_bars)
                bars = all_bars[bar_start_idx:bar_end_idx]
                # flatten
                if bars != []:
                    bars = torch.cat(bars, dim=0)
                    symbols = tuples_to_str(bars.cpu().numpy(), self.maps["idx2event"])
                    n_instruments = get_n_instruments(symbols)
                else:
                    n_instruments = 0

                j += 1
            if n_instruments < self.min_n_instruments:
                return None, None, None

            # transpose
            if self.transpose_options != []:
                n_transpose = random.choice(self.transpose_options)
                bars = transpose(bars, n_transpose, 
                                self.maps["transposable_event_inds"])

            # convert to indices (final input)
            bars = tensor_to_ind_tensor(bars, self.maps["tuple2idx"])

            # Decide taking the sample from the start of a bar or not
            r = np.random.uniform()

            start_at_beginning = not (r > self.bar_start_prob and bars.size(0) > self.input_len)
            
            if start_at_beginning:   
                # starts exactly at bar location
                if self.start_token is not None:
                    # add start token
                    start_idx = torch.ShortTensor(
                        [self.maps["tuple2idx"][self.start_token]])
                    bars = torch.cat((start_idx, bars), dim=0)
            else:
                # it doesn't have to start at bar location so shift arbitrarily
                start = np.random.randint(0, bars.size(0)-self.input_len)
                bars = bars[start:start+self.input_len+1]

            if self.regression and self.use_cls_token:
                # prepend <CLS> token
                cls_idx = torch.ShortTensor(
                    [self.maps["tuple2idx"][self.cls_token]])
                bars = torch.cat((cls_idx, bars), 0)

            # for now, no auxiliary conditions
            condition = torch.FloatTensor([np.nan, np.nan])
            if self.conditioning == "discrete_token" and \
                (start_at_beginning or self.always_use_discrete_condition):
                # add emotion tokens
                valence, arousal = self.data[idx]["valence"], self.data[idx]["arousal"]
                valence = torch.ShortTensor([self.maps["tuple2idx"][valence]])
                arousal = torch.ShortTensor([self.maps["tuple2idx"][arousal]])
                bars = torch.cat((valence, arousal, bars), dim=0)
            elif self.conditioning in ("continuous_token", "continuous_concat") or self.regression:
                # continuous conditions
                condition = torch.FloatTensor([self.data[idx]["valence"], self.data[idx]["arousal"]])
                
            bars = bars[:self.input_len + 1]    # trim to length, +1 to include target

            if self.pad_token is not None:
                n_pad = self.input_len + 1 - bars.shape[0]
                if n_pad > 0:
                    # pad if necessary
                    bars = torch.nn.functional.pad(bars, (0, n_pad), value=self.get_pad_idx()) 
            
            bars = bars.long()  # to int32
            input_ = bars[:-1]

            if self.regression:
                target = None   # will use condition as target
            else:
                target = bars[1:]
                if self.conditioning == "continuous_token":
                    # pad target from left, because input will get conditions concatenated
                    # their sizes should match
                    target = torch.nn.functional.pad(target, (condition.size(0), 0), value=self.get_pad_idx()) 
            
            if self.overfit:
                self.one_sample = [input_, condition, target]
        else:
            # sanity check, using one sample repeatedly
            input_, condition, target = self.one_sample

        return input_, condition, target


    

        

        




