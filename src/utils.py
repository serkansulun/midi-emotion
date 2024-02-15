import torch
import csv
import shutil
import functools
import os
import nvsmi
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

def plot_performance(csv_path, start_step=1, title=None, save=True):

    keys = [
        "trn_loss", 
        "val_loss", 
        # "map_macro", 
        ]
    data = read_csv(csv_path, numeric=True)
    x_lr_changes = []
    vals = {key: {"x":[], "y":[]} for key in keys}
    old_lr = data[0]["lr"]
    for item in data:
        step = item["step"]
        if step >= start_step:
            new_lr = item["lr"]
            if new_lr < old_lr:
                x_lr_changes.append(step)
                old_lr = new_lr

            for key in keys:
                val = item[key]
                if not np.isnan(val):
                    vals[key]["x"].append(step)
                    vals[key]["y"].append(val)
    plt.figure(dpi=300)
    for key, points in vals.items():
        plt.plot(points["x"], points["y"], label=key)

    label = f"LR changes (x{len(x_lr_changes)})"
    for x in x_lr_changes:
        plt.axvline(x=x, color="black", linestyle="--", linewidth=1, label=label)
        label = None
    plt.legend()
    plt.grid()
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    if title == None:
        title = csv_path.split("/")[-2]
    plt.title(title)
    png_path = csv_path.replace(".csv", ".pdf")
    if save:
        plt.savefig(png_path)
    else:
        plt.show()
    plt.close()

def memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() // 1024**2
        max_allocated = torch.cuda.max_memory_allocated() // 1024**2
        cached = torch.cuda.memory_reserved() // 1024**2
        max_cached = torch.cuda.max_memory_reserved() // 1024**2
        # print(list(nvsmi.get_gpus()))
        gpu = list(nvsmi.get_gpus())[0]
        nvidia_used = gpu.mem_used
        mem_dict = {
            "allocated": allocated,
            "max_allocated": max_allocated,
            "cached": cached,
            "max_cached": max_cached,
            "nvsmi": nvidia_used
        }
        mem_str = ", ".join([f"{key}: {val:.0f}" for key, val in mem_dict.items()])
    else:
        mem_str = "CUDA is not available."
    return mem_str


def split_list(alist, n_parts):
    if n_parts == 0:
        n_parts = 1
    length = len(alist)
    return [ alist[i*length // n_parts: (i+1)*length // n_parts] 
            for i in range(n_parts)]

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5), ignore_index=None):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/3

    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
                
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        
        output = output.reshape(-1, output.size(-1))
        target = target.reshape(-1)

        valid_inds = torch.where(target != ignore_index)[0]
        target = target[valid_inds]
        output = output[valid_inds, :]
        
        sample_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=-1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = {}
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / sample_size  # topk accuracy for entire batch
            list_topk_accs[k] = topk_acc.item()
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

class CsvWriter:
    # Save performance as a csv file
    def __init__(self, out_path, fieldnames, in_path=None, debug=False):

        self.out_path = out_path
        self.fieldnames = fieldnames
        self.debug = debug

        if not debug:
            if in_path is None:
                with open(out_path, "w") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
            else:
                try:
                    shutil.copy(in_path, out_path)
                except:
                    with open(out_path, "w") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()


    def update(self, performance_dict):
        if not self.debug:
            with open(self.out_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(performance_dict)
            a = 0
    
def generate_square_subsequent_mask(sz):
    # Triangular mask to avoid looking at future tokens
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def logging(s, log_path, print_=True, log_=True):
    # Prints log
    if print_:
        print(s, flush=True)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(dir_path, debug=False):
    # Create experiment directory
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)
    else:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        print('Experiment dir : {}'.format(dir_path))

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))


def get_n_instruments(symbols):
    # Find number of instruments
    symbols_split = [s.split("_") for s in symbols]
    symbols_split = [s[1] for s in symbols_split if len(s) == 3]
    events = list(set(symbols_split))
    return len(events)

def read_csv(input_file_path, delimiter=",", numeric=False):
    with open(input_file_path, "r") as f_in:
        reader = csv.DictReader(f_in, delimiter=delimiter)
        if numeric:
            data = [{key: float(value) for key, value in row.items()} for row in reader]
        else:
            data = [{key: value for key, value in row.items()} for row in reader]
    return data