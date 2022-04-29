import pretty_midi
import csv

# For reverse processing (TOKENS TO MIDI)

def tensor_to_tuples(x):
    x = x.tolist()
    x = [tuple(el) for el in x]
    return x


def tuples_to_mid(x, idx2event, verbose=False):
    # Tuples to midi
    instrument_to_program = {"DRUMS": (0, True), "PIANO": (0, False), "GUITAR": (24, False), 
                             "BASS": (32, False), "STRINGS": (48, False)}
    velocities = {
    "BASS": 127,
    "DRUMS": 120,
    "GUITAR": 95,
    "PIANO": 110,
    "STRINGS": 85,
    }

    tracks = {}
    for key, val in instrument_to_program.items():
        track = pretty_midi.Instrument(program=val[0], is_drum=val[1], name=key.lower())
        track.notes = []
        tracks.update({key: track})

    active_notes = {}

    time_cursor = 0
    for el in x:
        if el[0] != "<":     # if not special token
            event = idx2event[el[0]]       
            if "TIMESHIFT" == event:
                timeshift = float(el[1])
                time_cursor += timeshift / 1000.0
            else:
                on_off, instrument = event.split("_")
                pitch = int(el[1])
                if on_off == "ON":
                    active_notes.update({(instrument, pitch): time_cursor})
                elif (instrument, pitch) in active_notes:
                    start = active_notes[(instrument, pitch)]
                    end = time_cursor
                    tracks[instrument].notes.append(pretty_midi.Note(velocities[instrument], pitch, start, end))
                elif verbose:  
                    print("Ignoring {:>15s} {:4} because there was no previos ""ON"" event".format(event, pitch))

    mid = pretty_midi.PrettyMIDI()
    mid.instruments += tracks.values()
    return mid


def ind_tensor_to_tuples(x, ind2tuple):
    # Indices to tuples
    x = [ind2tuple[el.item()] for el in x]
    return x

def tuples_to_str(x, idx2event):
    # Tuples to strings
    str_list = []
    for el in x:
        if el[0] == "<":    # special token
            str_list.append(el)
        else:
            str_list.append(idx2event[el[0]] + "_" + str(el[1]))
    return str_list

def ind_tensor_to_mid(x, idx2tuple, idx2event, verbose=False):
    # Indices to midi
    x = ind_tensor_to_tuples(x, idx2tuple)
    x = tuples_to_mid(x, idx2event, verbose=verbose)
    return x

def ind_tensor_to_str(x, idx2tuple, idx2event):
    # Indices to string
    x = ind_tensor_to_tuples(x, idx2tuple)
    x = tuples_to_str(x, idx2event)
    return x