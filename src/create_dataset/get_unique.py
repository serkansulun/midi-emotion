import json
import pretty_midi
import pypianoroll
import hdf5_getters
from tqdm import tqdm
import os
import concurrent.futures
import collections
import utils
from glob import glob
import pandas as pd

def run_parallel(func, my_iter):
    # Parallel processing visualized with tqdm
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, my_iter), total=len(my_iter)))
    return results

write = False
redo = False

main_output_dir = "../../data_files/features"
os.makedirs(main_output_dir, exist_ok=True)

match_scores_path = "../../data_files/match_scores.json"
msd_summary_path = "../../data_files/msd_summary_file.h5"
echonest_folder_path = "../../data_files/millionsongdataset_echonest"

use_pianoroll_dataset = True
if use_pianoroll_dataset:
    midi_dataset_path = "../../data_files/lpd_5/lpd_5_full"
    extension = ".npz"
    output_dir = os.path.join(main_output_dir, "pianoroll")
else:
    midi_dataset_path = "lmd_full"
    extension = ".mid"
    output_dir = os.path.join(main_output_dir, "midi")
os.makedirs(output_dir, exist_ok=True)

# ### PART I: Map track_ids (in midi dataset) to Spotify features

# ### 1- Create mappings track_id (in midi dataset) -> metadata (in Echonest)

# output_path = os.path.join(output_dir, "trackid_to_songid.json")

# with open(match_scores_path, "r") as f:
#     match_scores = json.load(f)

# track_ids = sorted(list(match_scores.keys()))

# if os.path.exists(output_path) and not redo:
#     with open(output_path, "r") as f:
#         trackid_to_songid = json.load(f)
# else:
#     h5_msd = hdf5_getters.open_h5_file_read(msd_summary_path)
#     n_msd = hdf5_getters.get_num_songs(h5_msd)

#     trackid_to_songid = {}
#     print("Adding metadata to each track in Lakh dataset")

#     for i in tqdm(range(n_msd)):
#         track_id = hdf5_getters.get_track_id(h5_msd, i).decode("utf-8")
#         if track_id in track_ids:
#             # get data from MSD
#             song_id = hdf5_getters.get_song_id(h5_msd, i).decode("utf-8")
#             artist = hdf5_getters.get_artist_name(h5_msd, i).decode("utf-8")
#             title = hdf5_getters.get_title(h5_msd, i).decode("utf-8")
#             release = hdf5_getters.get_release(h5_msd, i).decode("utf-8")
#             trackid_to_songid[track_id] = {"song_id": song_id,"title": title, 
#                             "artist": artist, "release": release}

#     # sort
#     trackid_to_songid = collections.OrderedDict(sorted(trackid_to_songid.items()))
#     if write:
#         with open(output_path, "w") as f:
#             json.dump(trackid_to_songid, f, indent=4)
#             print(f"Output saved to {output_path}")

# ### 2- Create mappings metadata (in Echonest) -> Spotify IDs
# output_path = os.path.join(output_dir, "songid_to_spotify.json")
# if os.path.exists(output_path) and not redo:
#     with open(output_path, "r") as f:
#         songid_to_spotify = json.load(f)
# else:
#     song_ids = sorted([val["song_id"] for val in trackid_to_songid.values()])
#     songid_to_spotify = {}
#     print("Mapping Echonest song IDs to Spotify song IDs")
#     for song_id in tqdm(song_ids):
#         file_path = os.path.join(echonest_folder_path, song_id[2:4], song_id + ".json")
#         spotify_ids = utils.get_spotify_ids(file_path)
#         songid_to_spotify[song_id] = spotify_ids
#     if write:
#         with open(output_path, "w") as f:
#             json.dump(songid_to_spotify, f, indent=4)
#             print(f"Output saved to {output_path}")


# ### 3- Merge and add Spotify features
# output_path = os.path.join(output_dir, "trackid_to_spotify_features.json")
# if os.path.exists(output_path) and not redo:
#     with open(output_path, "r") as f:
#         trackid_to_spotify_features = json.load(f)
# else:
#     print("Adding Spotify features")
#     trackid_to_spotify_features = {}
#     for track_id, data in tqdm(trackid_to_songid.items()):
#         try:
#             album = data["release"]
#             spotify_ids = songid_to_spotify[data["song_id"]]
#             if spotify_ids == []:
#                 # use metadata to search spotify
#                 best_spotify_track = utils.search_spotify_flexible(data["title"], data["artist"], data["release"])
#             else:
#                 spotify_tracks = utils.get_spotify_tracks(spotify_ids)
#                 if len(spotify_tracks) > 1:
#                     # find best spotify id by comparing album names
#                     best_match_score = 0
#                     best_match_ind = 0
#                     for i, track in enumerate(spotify_tracks):
#                         if track is not None:
#                             spotify_album = track["album"]["name"] if track is not None else ""
#                             match_score = utils.matching_strings_flexible(album, spotify_album)
                            
#                             if match_score > best_match_score:
#                                 best_match_score = match_score
#                                 best_match_ind = i

#                     best_spotify_track = spotify_tracks[best_match_ind]
#                 else:
#                     best_spotify_track = spotify_tracks[0]
            
#             if best_spotify_track is not None:
#                 spotify_id = best_spotify_track["uri"].split(":")[-1]
#                 spotify_audio_features = utils.get_spotify_features(spotify_id)[0]

#                 # if spotify_audio_features["valence"] == 0.0:
#                 #     # A large portion of files have 0.0 valence, although they are NaNs
#                 #     spotify_audio_features["valence"] = float("nan")
#                 spotify_artists = ", ".join([artist["name"] for artist in best_spotify_track["artists"]])

#                 data["spotify_id"] = spotify_id
#                 data["spotify_title"] = best_spotify_track['name']
#                 data["spotify_artist"] = spotify_artists
#                 data["spotify_album"] = best_spotify_track["album"]["name"]
#                 data["spotify_audio_features"] = spotify_audio_features
#             else:
#                 for key in ["id", "title", "artist", "album", "audio_features"]:
#                     data["spotify_" + key] = None

#             trackid_to_spotify_features[track_id] = data
#         except:
#             print(f"Problematic track: {track_id}")
#     if write:
#         with open(output_path, "w") as f:
#             json.dump(trackid_to_spotify_features, f, indent=4)


# ### PART II: Dealing with symbolic music data
# ### 4- Revert matching scores
# """ Matched data has the format: track_ID -> midi_file 
# where multiple tracks could be mapped to a single midi file.
# We want to revert this mapping and then keep unique midi files
# Revert match scores file to have mapping midi_file -> track_ID
# """

# output_path = os.path.join(output_dir, "match_scores_reverse.json")

# if os.path.exists(output_path) and not redo:
#     with open(output_path, "r") as f:
#         match_scores_reversed = json.load(f)
# else:
#     with open(match_scores_path, "r") as f:
#         in_data = json.load(f)

#     match_scores_reversed = {}
#     print("Reversing match scores")
#     for track_id, matching in tqdm(in_data.items()):
#         for file_, score in matching.items():
#             if file_ not in match_scores_reversed.keys():
#                 match_scores_reversed[file_] = {track_id: score}
#             else:
#                 match_scores_reversed[file_][track_id] = score

#     # order match scores
#     for k in match_scores_reversed.keys():
#         match_scores_reversed[k] = collections.OrderedDict(sorted(match_scores_reversed[k].items(), reverse=True, key=lambda x: x[-1]))

#     # order filenames
#     match_scores_reversed = collections.OrderedDict(sorted(match_scores_reversed.items(), key=lambda x: x[0]))
#     if write:
#         with open(output_path, "w") as f:
#             json.dump(match_scores_reversed, f, indent=4)

# # 5- Filter match scores to only keep best match

# output_path = os.path.join(output_dir, "best_match_scores.json")
# if os.path.exists(output_path) and not redo:
#     with open(output_path, "r") as f:
#         best_match_scores_reversed = json.load(f)
# else:
#     best_match_scores_reversed = {}
#     for midi_file, match in tqdm(match_scores_reversed.items()):
#         best_match_scores_reversed[midi_file] = list(match.items())[0]
#     if write:
#         with open(output_path, "w") as f:
#             json.dump(best_match_scores_reversed, f, indent=4)

### 6- Filter unique midis
"""LMD was created by creating hashes for the entire files
and then keeping files with unique hashes.
However, some files' musical content are the same, and only their metadata are different.
So we hash the content (pianoroll array), and further filter out the unique ones."""
# Create hashes for midis

output_path = os.path.join(output_dir, "hashes.json")

if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_file_to_hash = json.load(f)
else:
    def get_hash_and_file(path):
        hash_ = utils.get_hash(path)
        file_ = os.path.basename(path)
        file_ = file_[:-4]
        return [file_, hash_]

    file_paths = sorted(glob(midi_dataset_path + "/**/*" + extension, recursive=True))

    print("Getting hashes for midis.")
    midi_file_to_hash = run_parallel(get_hash_and_file, file_paths)
    midi_file_to_hash = sorted(midi_file_to_hash, key=lambda x:x[0])
    midi_file_to_hash = dict(midi_file_to_hash)
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_file_to_hash, f, indent=4)
            print(f"Output saved to {output_path}")

# also do the reverse hash -> midi
output_path = os.path.join(output_dir, "unique_files.json")
if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_files_unique = json.load(f)
else:
    hash_to_midi_file = {}
    for midi_file, hash in midi_file_to_hash.items():
        try:
            best_match_score = best_match_scores_reversed[midi_file][1]
        except:
            best_match_score = 0
        if hash in hash_to_midi_file.keys():
            hash_to_midi_file[hash].append((midi_file, best_match_score))
        else:
            hash_to_midi_file[hash] = [(midi_file, best_match_score)]

    midi_files_unique = []
    # Get unique midis (with highest match score)
    for hash, midi_files_and_match_scores in hash_to_midi_file.items():
        if hash != "empty_pianoroll":
            midi_files_and_match_scores = sorted(midi_files_and_match_scores, key=lambda x: x[1], reverse=True)
            midi_files_unique.append(midi_files_and_match_scores[0][0])
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_files_unique, f, indent=4)

# create unique matched midis list
midi_files_matched = list(match_scores_reversed.keys())

output_path = os.path.join(output_dir, "midis_matched_unique.json")
if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_files_matched_unique = json.load(f)
else:
    midi_files_matched_unique = sorted(list(set(midi_files_matched).intersection(midi_files_unique)))
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_files_matched_unique, f, indent=4)

# create unique unmatched midis list
output_path = os.path.join(output_dir, "midis_unmatched_unique.json")
if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_files_unmatched_unique = json.load(f)
else:
    midi_files_unmatched_unique = sorted(list(set(midi_files_unique) - set(midi_files_matched_unique)))
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_files_unmatched_unique, f, indent=4)

### 6- Create mappings: midi -> best matching track ID, spotify features
output_path = os.path.join(output_dir, "spotify_features.json")
if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_file_to_spotify_features = json.load(f)
else:
    midi_file_to_spotify_features = {}
    for pr in tqdm(midi_files_matched_unique):
        sample_data = {}
        sample_data["track_id"], sample_data["match_score"] = best_match_scores_reversed[pr]
        metadata_and_spotify = trackid_to_spotify_features[sample_data["track_id"]]
        sample_data.update(metadata_and_spotify)
        midi_file_to_spotify_features[pr] = sample_data
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_file_to_spotify_features, f, indent=4)

### 7- For all midis, get low level features 
# (tempo, note density, number of instruments)

output_path = os.path.join(output_dir, "midi_features.json")
if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_file_to_midi_features = json.load(f)
else:
    def get_midi_features(midi_file):
        midi_path = os.path.join(midi_dataset_path, midi_file[0], midi_file + extension)
        if use_pianoroll_dataset:
            mid = pypianoroll.load(midi_path).to_pretty_midi()
        else:
            mid = pretty_midi.PrettyMIDI(midi_path)
        note_density = utils.get_note_density(mid)
        tempo = utils.get_tempo(mid)
        n_instruments = utils.get_n_instruments(mid)
        duration = mid.get_end_time()
        midi_features = {
            "note_density": note_density,
            "tempo": tempo,
            "n_instruments": n_instruments,
            "duration": duration,
        }
        return [midi_file, midi_features]

    midi_file_to_midi_features = run_parallel(get_midi_features, midi_files_unique)
    midi_file_to_midi_features = dict(midi_file_to_midi_features)
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_file_to_midi_features, f, indent=4)

### 8- Merge midi features and matched (Spotify) features

output_path = os.path.join(output_dir, "full_dataset_features.json")
if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        midi_file_to_merged_features = json.load(f)
else:
    midi_file_to_merged_features = {}
    for midi_file in tqdm(midi_file_to_midi_features.keys()):
        midi_file_to_merged_features[midi_file] = {}
        midi_file_to_merged_features[midi_file]["midi_features"] = midi_file_to_midi_features[midi_file]
        if midi_file in midi_file_to_spotify_features.keys():
            matched_features = midi_file_to_spotify_features[midi_file]
        else:
            matched_features = {}
        midi_file_to_merged_features[midi_file]["matched_features"] = matched_features
    if write:
        with open(output_path, "w") as f:
            json.dump(midi_file_to_merged_features, f, indent=4)

### Do the same for matched dataset
output_path = os.path.join(output_dir, "matched_dataset_features.json")
if os.path.exists(output_path) and not redo:
    with open(output_path, "r") as f:
        matched_midi_file_to_merged_features = json.load(f)
else:
    matched_midi_file_to_merged_features = \
        {file_: midi_file_to_merged_features[file_] for file_ in tqdm(midi_files_matched_unique)}
    if write:
        with open(output_path, "w") as f:
            json.dump(matched_midi_file_to_merged_features, f, indent=4)

### PART III: Constructing training dataset
### 9- Summarize matched dataset features by only taking valence and note densities per instrument,
# number of instruments, durations, is_matched

output_path = os.path.join(output_dir, "full_dataset_features_summarized.csv")
if not os.path.exists(output_path) or redo:
    dataset_summarized = []
    for midi_file, features in tqdm(midi_file_to_merged_features.items()):
        
        midi_features = features["midi_features"]
        n_instruments = midi_features["n_instruments"]
        note_density_per_instrument = midi_features["note_density"] / n_instruments
        
        matched_features = features["matched_features"]
        if matched_features == {}:
            is_matched = False
            valence = float("nan")
        else:
            is_matched = True
            spotify_audio_features = matched_features["spotify_audio_features"]
            if spotify_audio_features is None or spotify_audio_features["valence"] == 0.0:
                # An unusual number of samples have a valence of 0.0
                # which is possibly due to an error. Feel free to comment out.
                valence = float("nan")
            else:
                valence = spotify_audio_features["valence"]
        
        dataset_summarized.append({
            "file": midi_file,
            "is_matched": is_matched,
            "n_instruments": n_instruments,
            "note_density_per_instrument": note_density_per_instrument,
            "valence": valence
        })

    dataset_summarized = pd.DataFrame(dataset_summarized)
        
    if write:
        dataset_summarized.to_csv(output_path, index=False)