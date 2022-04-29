import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re
import hashlib
import json
import pypianoroll
import numpy as np
import pretty_midi

"""
You'll need a client ID and a client secret:
https://developer.spotify.com/dashboard/applications
Then, fill in the variables client_id and client_secret
"""

client_id = 'to fill in'
client_secret = 'to fill in'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_drums_note_density(mid):
    drum_mid = pretty_midi.PrettyMIDI()
    for instrument in mid.instruments:
        if instrument.is_drum:
            drum_mid.instruments.append(instrument)
    if len(drum_mid.instruments) != 1 or len(drum_mid.instruments[0].notes) == 0:
        return float("nan")
    else:
        start_time = drum_mid.instruments[0].notes[0].start
        end_time = drum_mid.instruments[0].notes[-1].end
        duration = end_time - start_time
        n_notes = len(drum_mid.instruments[0].notes)
        density = n_notes / duration
        return density

def get_md5(path):
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return md5

def get_hash(path):
    if path[-4:] == ".mid":
        try:
            mid = pretty_midi.PrettyMIDI(path)
        except:
            return "empty_pianoroll"
        try:
            rolls = mid.get_piano_roll()
        except:
            return "empty_pianoroll"
        if rolls.size == 0:
            return "empty_pianoroll"
    else:
        pr = pypianoroll.load(path)
        tracks = sorted(pr.tracks, key=lambda x: x.name)
        rolls = [track.pianoroll for track in tracks if track.pianoroll.shape[0] > 0]
        if rolls == []:
            return "empty_pianoroll"
        rolls = np.concatenate(rolls, axis=-1)
    hash_ = hashlib.sha1(np.ascontiguousarray(rolls)).hexdigest()
    return hash_

def get_note_density(mid):
    duration = mid.get_end_time()
    n_notes = sum([1 for instrument in mid.instruments for note in instrument.notes])
    density = n_notes / duration
    return density

def get_tempo(mid):
    tick_scale = mid._tick_scales[-1][-1]
    resolution = mid.resolution
    beat_duration = tick_scale * resolution
    mid_tempo = 60 / beat_duration
    return mid_tempo

def get_n_instruments(mid):
    n_instruments = sum([1 for instrument in mid.instruments if instrument.notes != []])
    return n_instruments

def try_multiple(func, *args, **kwargs):
    n_max = 29
    n = 0
    failed = True
    while failed:
        if n > n_max:
            return None
        try:
            if args:
                out = func(*args)
            elif kwargs:
                out = func(**kwargs)
            failed = False
        except Exception as e:
            if e.args[0] == 404:
                return None
            else:
                n += 1
    return out

def search_spotify(title, artist, album=None):
    query = '"{}"+artist:"{}"'.format(title, artist)
    if album is not None:
        query += '+album:"{}"'.format(album)
    result = try_multiple(sp.search, q=query, type='track')
    items = result['tracks']['items']
    return items


def search_spotify_flexible(title, artist, album):
    # Find Spotify URI based on metadata
    items = search_spotify(title, artist, album)
    if items == []:
        items = search_spotify(title, artist)
    if items == []:
        title = fix_string(title)
        items = search_spotify(title, artist)
    if items == []:
        artist = fix_string(artist)
        items = search_spotify(title, artist)
    if items == []:
        artist = strip_artist(artist)
        items = search_spotify(title, artist)
    if items == []:
        return None

    elif len(items) == 1:
        item = items[0]
    else:
        # Return most popular
        max_popularity = 0
        best_ind = 0
        for i, item in enumerate(items):
            if item["popularity"] > max_popularity:
                max_popularity = item["popularity"]
                best_ind = i
        item = items[best_ind]
    return item

def matching_strings_flexible(a, b):
    if a == "" or b == "":
        matches = 0.0
    else:
        a = fix_string(a)
        b = fix_string(b)
        a = a.replace("'", "")
        b = b.replace("'", "")
        min_len = min(len(a), len(b))
        matches = 0
        for i in range(min_len):
            if a[i] == b[i]:
                matches += 1
        matches /= min_len
    return matches

def get_spotify_features(uri_list):
    features = try_multiple(sp.audio_features, uri_list)
    return features

def get_spotify_tracks(uri_list):
    if len(uri_list) > 50:
        uri_list = uri_list[:50]
    tracks = try_multiple(sp.tracks, uri_list)
    return tracks["tracks"]


def strip_artist(s):
    s = s.lower()   # lowercase
    s = s.replace("the ", "")
    keys = [' - ', '/', ' ft', 'feat', 'featuring', ' and ', ' with ', '_', ' vs', '&', ';', '+']
    for key in keys:
        loc = s.find(key)
        if loc != -1:
            s = s[:loc]
    return s

def fix_string(s):
    if s != "":
        s = s.lower()   # lowercase
        s = s.replace('\'s', '')    # remove 's
        s = s.replace('_', ' ')    # remove _
        s = re.sub("[\(\[].*?[\)\]]", "", s)    # remove everything in parantheses
        if s[-1] == " ":    # remove space at the end
            s = s[:-1]
    return s

def logprint(s, f):
    f.write(s + '\n')

def get_spotify_ids(json_path):
    with open(json_path) as f_json:
        json_data = json.load(f_json)
        json_data = json_data["response"]["songs"]
        if len(json_data) == 0:
            spotify_ids = []
        else:
            json_data = json_data[0]
            spotify_ids = []
            for track in json_data["tracks"]:
                if track["catalog"] == "spotify" and "foreign_id" in list(track.keys()):
                    spotify_ids.append(track["foreign_id"].split(":")[-1])
    return spotify_ids