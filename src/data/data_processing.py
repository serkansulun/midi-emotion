import pypianoroll
from operator import attrgetter
import torch
from copy import deepcopy
import numpy as np

# Forward processing. (Midi to indices)

def read_pianoroll(fp, return_tempo=False):
    # Reads pianoroll file and converts to PrettyMidi
    pr = pypianoroll.load(fp)
    mid = pr.to_pretty_midi()
    if return_tempo:
        tempo = np.mean(pr.tempo)
        return mid, tempo
    else:
        return mid

def trim_midi(mid_orig, start, end, strict=True):
    """Trims midi file

    Args:
        mid (PrettyMidi): input midi file
        start (float): start time
        end (float): end time
        strict (bool, optional): 
            If false, includes notes that starts earlier than start time,
            and ends later than start time. Or ends later than end time,
            but starts earlier than end time. The start and end times 
            are readjusted so they fit into the given boundaries.
            Defaults to True.

    Returns:
        (PrettyMidi): Trimmed output MIDI.
    """
    eps = 1e-3
    mid = deepcopy(mid_orig)
    for ins in mid.instruments:
        if strict:
            ins.notes = [note for note in ins.notes if note.start >= start and note.end <= end]
        else:
            ins.notes = [note for note in ins.notes \
                 if note.end > start + eps and note.start < end - eps]

        for note in ins.notes:
            if not strict:
                # readjustment
                note.start = max(start, note.start)
                note.end = min(end, note.end)
            # Make the excerpt start at time zero
            note.start -= start
            note.end -= start
    # Filter out empty tracks
    mid.instruments = [ins for ins in mid.instruments if ins.notes]
    return mid


def mid_to_timed_tuples(music, event_sym2idx, min_pitch: int = 21, max_pitch: int = 108):
    # for sorting (though not absolutely necessary)
    on_off_priority = ["ON", "OFF"]
    ins_priority = ["DRUMS", "BASS", "GUITAR", "PIANO", "STRINGS"]

    on_off_priority = {val: i for i, val in enumerate(on_off_priority)}
    ins_priority = {val: i for i, val in enumerate(ins_priority)}

    # Add instrument info to notes
    for i, track in enumerate(music.instruments):
        for note in track.notes:
            note.instrument = track.name

    # Collect notes
    notes = []
    for track in music.instruments:
        notes.extend(track.notes)

    # Raise an error if no notes is found
    if not notes:
        raise RuntimeError("No notes found.")

    # Sort the notes
    notes.sort(key=attrgetter("start", "pitch", "duration", "velocity", "instrument"))

    # Collect note-related events
    note_events = []

    for note in notes:
        if note.pitch >= min_pitch and note.pitch <= max_pitch:

            start = round(note.start, 6)
            end = round(note.end, 6)

            ins = note.instrument.upper()

            note_events.append((start, on_off_priority["ON"],
                ins_priority[ins], (event_sym2idx["_".join(["ON", ins])], note.pitch)))
            note_events.append((end, on_off_priority["OFF"],
                ins_priority[ins], (event_sym2idx["_".join(["OFF", ins])], note.pitch)))

    # Sort events by time
    note_events = sorted(note_events)
    note_events = [(note[0], note[-1]) for note in note_events]
    return note_events

def timed_tuples_to_tuples(note_events, event_sym2idx, max_timeshift: int = 1000, 
    timeshift_step: int = 8):

    # Create a list for all events
    events = []
    # Initialize the time cursor
    time_cursor = int(round(note_events[0][0] * 1000))
    # Iterate over note events
    for time, symbol in note_events:
        time = int(round(time * 1000))
        if time > time_cursor:
            timeshift = time - time_cursor
            # First split timeshifts longer than max
            n_max = timeshift // max_timeshift
            for _ in range(n_max):
                events.append((event_sym2idx["TIMESHIFT"], max_timeshift))
            # quantize and add remaining
            rem = timeshift % max_timeshift
            if rem > 0:
                # do not round to zero
                rem = int(timeshift_step * round(float(rem) / timeshift_step))
                if rem == 0:
                    rem = timeshift_step    # do not round to zero
                events.append((event_sym2idx["TIMESHIFT"], rem))
            time_cursor = time
        if symbol[0] != "<":    # if not special symbol
            events.append(symbol)
    return events


def list_to_tensor(list_, sym2idx):
    indices = [sym2idx[sym] for sym in list_]
    indices = torch.LongTensor(indices)
    return indices


def mid_to_bars(mid, event_sym2idx):
    """Takes MIDI, extracts bars
    returns ndarray where each row is a token
    each token has two elements, 
    first is an index of event, such as DRUMS_OFF, or TIMESHIFT
    second is the value (pitch for note or time for timeshift)
    """
    try:
        bar_times = [round(bar, 6) for bar in mid.get_downbeats()]
        bar_times.append(bar_times[-1] + (bar_times[-1] - bar_times[-2]))   # to end
        bar_times.append(bar_times[-1] + (bar_times[-1] - bar_times[-2]))   # to end

        note_events = mid_to_timed_tuples(mid, event_sym2idx)
        i_bar = -1
        i_note = 0
        bars = []
        cur_bar_note_events = []

        cur_bar_end = -float("inf")
        while i_note < len(note_events):
            time, note = note_events[i_note]
            if time < cur_bar_end:
                cur_bar_note_events.append((time, note))
                i_note += 1
            else:
                cur_bar_note_events.append((cur_bar_end, "<BAR_END>"))
                if len(cur_bar_note_events) > 2:
                    events = timed_tuples_to_tuples(cur_bar_note_events, event_sym2idx)
                    events = tuples_to_array(events)
                    bars.append(events)
                i_bar += 1
                cur_bar_start = bar_times[i_bar]
                cur_bar_end = bar_times[i_bar+1]
                cur_bar_note_events = [(cur_bar_start, "<BAR_START>")]
    except:
        bars = None
    return bars

def tuples_to_array(x):
    x = [list(el) for el in x]
    x = np.asarray(x, dtype=np.int16)
    return x

def get_maps(min_pitch=21,max_pitch=108,max_timeshift=1000,timeshift_step=8):
    # Get mapping dictionary
    instruments = ["DRUMS", "GUITAR", "BASS", "PIANO", "STRINGS"]
    special_symbols = ["<PAD>", "<START>"]
    on_offs = ["OFF", "ON"]

    token_syms = deepcopy(special_symbols)
    event_syms = []
    transposable_event_syms = []

    for ins in instruments:
        for on_off in on_offs:
            event_syms.append(f"{on_off}_{ins}")
            if ins != "DRUMS":
                transposable_event_syms.append(f"{on_off}_{ins}")
            for pitch in range(min_pitch, max_pitch + 1):
                token_syms.append((f"{on_off}_{ins}", pitch))
                
    for timeshift in range(timeshift_step, max_timeshift + timeshift_step, timeshift_step):
        token_syms.append(("TIMESHIFT", timeshift))
    event_syms.append("TIMESHIFT")

    map = {}
    
    map["event2idx"] = {sym: idx for idx, sym in enumerate(event_syms)}
    map["idx2event"] = {idx: sym for idx, sym in enumerate(event_syms)}

    map["tuple2idx"] = {}
    map["idx2tuple"] = {}
    for idx, sym in enumerate(token_syms):
        if isinstance(sym, tuple):
            indexed_tuple = (map["event2idx"][sym[0]], sym[1])
        else:
            indexed_tuple = sym
        map["tuple2idx"][indexed_tuple] = idx
        map["idx2tuple"][idx] = indexed_tuple

    transposable_event_inds = [map["event2idx"][sym] for sym in transposable_event_syms]
    map["transposable_event_inds"] = transposable_event_inds
    return map


def transpose(x, n, transposable_event_inds, min_pitch = 21, max_pitch = 108):
    # Transpose melody
    for i in range(x.size(0)):
        if x[i, 0].item() in transposable_event_inds and \
            x[i, 1].item() + n <= max_pitch and \
            x[i, 1].item() + n >= min_pitch:
            x[i, 1] += n
    return x

def tuples_to_ind_tensor(x, tuple2idx):
    # Tuples to indices
    x = [tuple2idx[el] for el in x]
    x = torch.tensor(x, dtype=torch.int16)
    return x

def tensor_to_tuples(x):
    x = [tuple(row.tolist()) for row in x]
    return x

def tensor_to_ind_tensor(x, tuple2idx):
    x = tensor_to_tuples(x)
    x = tuples_to_ind_tensor(x, tuple2idx)
    return x