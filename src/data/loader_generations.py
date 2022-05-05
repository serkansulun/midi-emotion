from glob import glob
import os
from tkinter import TRUE
import torch
import sys
sys.path.append("..")

"""
Data loader to perform regression on a folder with generations
"""

class LoaderGenerations:

    def __init__(self, gen_folder, seq_len, pad=True, use_start_token=True, use_end_token=False, 
                use_cls_token=TRUE, overlap=0.5):

        self.seq_len = seq_len
        self.one_sample = None

        self.pad = pad

        self.pad_token = '<PAD>' if pad else None
        self.start_token = '<START>' if use_start_token else None
        self.end_token = '<END>' if use_end_token else None
        self.cls_token = "<CLS>" if use_cls_token else None

        data_paths = glob(os.path.join("../output", gen_folder, "*.pt"), recursive=True)

        maps = torch.load("../datasets/lpd_5/w_emotion_transposable/maps.pt")
        n_vocab = len(maps["tuple2idx"])

        self.data = []

        if self.cls_token is not None:
            seq_len -= 1
            if self.cls_token not in maps["tuple2idx"].keys():
                # add <CLS> token to vobac
                maps["tuple2idx"][self.cls_token] = len(maps["idx2tuple"])
                maps["idx2tuple"][len(maps["idx2tuple"])] = self.cls_token
            # prepend <CLS> token
            cls_idx = torch.ShortTensor(
                [maps["tuple2idx"][self.cls_token]])

        for data_path in data_paths:
            generation = torch.load(data_path)
            inds = generation["inds"]
            # remove special tokens
            inds = inds[inds < n_vocab]              
            # split with overlap
            inds = inds.unfold(0, seq_len, int(seq_len*(1-overlap)))
            inds = list(torch.split(inds, 1, dim=0))
            inds = [sample.squeeze() for sample in inds]

            if self.cls_token is not None:
                inds = [torch.cat((cls_idx, sample), dim=0) for sample in inds]

            condition = generation["condition"]
            if inds[-1].size(0) != seq_len:
                inds.pop()
            self.data += [(sample, condition) for sample in inds]


        self.discrete2continuous = {
            "-2": -0.8,
            "-1": -0.4,
            "0": 0,
            "1": 0.4,
            "2": 0.8
        }


    def get_vocab_len(self):
        return None

    def get_maps(self):
        return None

    def get_pad_idx(self):
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        input_, condition = self.data[idx]
        if input_.size(0) != self.seq_len:
            Warning(f"Input length is {input_.size(0)}")
            return None, None, None
        if isinstance(condition[0], str):
            condition = condition[:2]
            for i in range(len(condition)):
                condition[i] = self.discrete2continuous[condition[i][2:-1]]
            condition = torch.Tensor(condition)

        input_ = input_.cpu()
        condition = condition.cpu()
        return input_, condition, None


        

        




