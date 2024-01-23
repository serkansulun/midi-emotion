import json
from data_processing import read_pianoroll, mid_to_bars, get_maps
import torch
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time
from functools import partial
import os
import argparse
import sys
sys.path.append('..')
import utils

""" Preprocessing Lakh MIDI pianoroll dataset.
Divides into bars. Encodes into tuples. Makes transposing easier. """

def run(f, my_iter):
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    return results

def get_emotion_dict(path):
    table = pd.read_csv(path)
    table = table.to_dict(orient="records")
    table = {item["path"].split("/")[-2]: \
                 {"valence": item["valence"], "energy": item["energy"], "tempo": item["tempo"]} \
                     for item in table}
    return table

def process(pr_path, event_sym2idx):
    time.sleep(0.001)
    mid = read_pianoroll(pr_path)

    bars = mid_to_bars(mid, event_sym2idx)

    file_ = pr_path.split("/")[-1]
    
    item_data = {
                "file": file_,
                "bars": bars, 
                 }

    return item_data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main_dir = "../../data_files/lpd_5"
    input_dir = "../../../../../../datasets/public/lakh_pianoroll_5"
    
    output_dir = os.path.join(main_dir, "lpd_5_full_transposable")
    
    os.makedirs(output_dir, exist_ok=True)
    output_maps_path = os.path.join(main_dir, "maps.pt")

    # # Get unique pianorolls
    # unique_pr_list_file = "../../data_files/features/pianoroll/unique_files.json"
    # with open(unique_pr_list_file, "r") as f:
    #     pr_paths = json.load(f)
    # pr_paths = [os.path.join(input_dir, pr_path[0], pr_path + ".npz") for pr_path in pr_paths]
    
    # Get unique pianorolls
    features_fp = "../../data_files/lpd_5/features.csv"
    features_data = utils.read_csv(features_fp)
    pr_paths = []
    for sample in features_data:
        id = sample['file']
        pr_path = os.path.join(input_dir, id[0], id + '.npz')
        pr_paths.append(pr_path)

    maps = get_maps()
    
    func = partial(process, event_sym2idx=maps["event2idx"])

    if args.debug:
        output_dir = output_dir + "_debug"
        pr_paths = pr_paths[:1000]
    os.makedirs(output_dir, exist_ok=True)

    x = run(func, pr_paths)
    x = [item for item in x if item["bars"] is not None]
    for i in tqdm(range(len(x))):
        for j in range(len(x[i]["bars"])):
            x[i]["bars"][j] = torch.from_numpy(x[i]["bars"][j])
        fname = x[i]["file"]
        output_path = os.path.join(output_dir, fname.replace(".npz", ".pt"))
        torch.save(x[i], output_path)

    torch.save(maps, output_maps_path)
    

if __name__ == "__main__":
    main()





