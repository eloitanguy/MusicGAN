"""
to_01 -> ignores velocity
no_silence -> delete silence at beginning / end
code_silence -> silence corresponds to [1,0,...,0]
"""

import mido
import string
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os


def load_midi(filename):
    """"
    Load midi file
    """
    midifolder = 'midi/'
    dotmid = '.mid'
    return mido.MidiFile(midifolder + filename + dotmid, clip=True)


"""
A MIDI file is a list of messages
"""


def msg2dict(msg):
    """
    Convert one message to one dictionary
    :param msg: midi message
    :return:
    """
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))

    return [result, on_]


def switch_note(last_state, note, velocity, on_=True, to_01=True, code_silence=True):
    """
    Map to [21,108] ie 88 notes on a piano
    """
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    add = int(code_silence)
    result = [0] * (88 + add) if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        if to_01:  # ADDED
            result[note - 21 + add] = 1 if on_ else 0
        else:
            result[note - 21 + add] = velocity if on_ else 0
    return result


def get_new_state(new_msg, last_state, to_01=True, code_silence=True):
    """
    For a single track
    """
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_, to_01=to_01,
                            code_silence=code_silence) if on_ is not None else last_state
    return [new_state, new_msg['time']]


def track2seq(track, to_01=True, code_silence=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    add = int(code_silence)
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0] * (88 + add), to_01=to_01, code_silence=code_silence)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state, to_01=to_01, code_silence=code_silence)
        if new_time > 0:
            result += [last_state] * new_time
        last_state, last_time = new_state, new_time
    return result


def mid2arry(mid, min_msg_pct=0.1, to_01=True, no_silence=True, code_silence=True, list_tracks=[0], s=24):
    """
    Handles multiple tracks, but usually we only want one track
    """
    add = int(code_silence)
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    if list_tracks != None:
        l = list_tracks
    else:
        l = range(len(mid.tracks))
    for i in l:
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i], to_01=to_01, code_silence=code_silence)
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * (88 + add)] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0, initial=-1)  # In case of overlap on different tracks
    if no_silence:
        # trim: remove consecutive 0s in the beginning and at the end
        sums = all_arys.sum(axis=1)
        ends = np.where(sums > 0)[0]
    else:
        ends = [0, all_arys.shape[0]]
    if code_silence:  # ADDED
        for i in range(all_arys.shape[0]):
            if np.max(all_arys[i, :]) == 0:
                all_arys[i, 0] = 1
    # Slice for lighter memory space
    return all_arys[min(ends):max(ends):s, :]


def keep(line, nb_keep, code_silence):
    """
    Filtering
    """
    if code_silence:
        temp = line[line > 0]
    else:
        temp = line

    if nb_keep == 0 or len(temp) == 0:
        return temp

    elif nb_keep == 1:
        return temp[[-1]]  # Highest note

    elif nb_keep == 2:
        return temp[[0]]  # Lowest note


def filter(arry, nb_keep=1):
    """
    Degree of "polyphony" kept
    0 -> no filtering
    1 -> keep top voice
    2 -> bottom voice
    """
    code_silence = arry.shape[1] == 88
    nozero = arry != 0
    collists = [keep(np.nonzero(t)[0], nb_keep, code_silence) for t in nozero]
    filtered = np.full(arry.shape, 0)
    for i, l in enumerate(collists):
        filtered[i, l] = arry[i, l]
    return filtered


def transposing(arry):
    """
    Returns a list of transposed tracks, including the original track
    """
    add = arry.shape[1] - 88
    res = []
    if add:
        pass
    else:
        nonzero_notes = np.nonzero(arry)[1]
        note_min, note_max = np.min(nonzero_notes), np.max(nonzero_notes)
        # 0 to 87
        for t in range(0, 88 - note_max + note_min):
            temp = np.full(arry.shape, 0)
            temp[:, t:(t + 1 + note_max - note_min)] = arry[:, note_min:(note_max + 1)]
            res.append(temp)
    return res


def combine(tracks):
    """
    Combining multiple tracks
    """
    # Column size is double the "piano" size
    add = tracks[0].shape[1] - 88
    all_arys = tracks
    # make all nested list the same length
    max_len = max([ary.shape[0] for ary in all_arys])
    for i in range(len(all_arys)):
        if all_arys[i].shape[0] < max_len:
            all_arys[i] = np.concatenate([all_arys[i], [([1] * add) + ([0] * 88)] * (max_len - len(all_arys[i]))])
    res = np.concatenate(all_arys, axis=1)
    return res


def plot_midi(p):
    fig = plt.figure()
    add = p.shape[1] - 88
    plt.plot(range(p.shape[0]), np.multiply(np.where(p[:, add:] > 0, 1, 0), range(1, 89)), marker='.', markersize=1,
             linestyle='')
    plt.show()


def arry2mid(arry, tempo=500000, velocity=100):
    """
    Back to MIDI
    """
    add = arry.shape[1] - 88
    ary = arry[:, add:]
    # get the difference
    new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
    changes = new_ary[1:] - new_ary[:-1]
    # create a midi file with an empty track
    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    mid_new.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    # add difference in the empty track
    last_time = 0
    for ch in changes:
        if set(ch) == {0}:  # no change
            last_time += 1
        else:
            on_notes = np.where(ch > 0)[0]
            on_notes_vol = ch[on_notes]
            off_notes = np.where(ch < 0)[0]
            first_ = True
            for n, v in zip(on_notes, on_notes_vol):
                new_time = last_time if first_ else 0
                if v == 1:  # ADDED
                    track.append(mido.Message('note_on', note=n + 21, velocity=velocity, time=new_time))
                else:
                    track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
                first_ = False
            for n in off_notes:
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
                first_ = False
            last_time = 0
    return mid_new


if __name__ == '__main__':
    names = open("songnames.txt", "r")

    for name in names.readlines():
        name = name.rstrip()
        if len(name) > 0:
            t0 = time()
            print(f"\nLoading song : {name}...")
            mid = load_midi(name)

            right_hand = mid2arry(mid, list_tracks=[0])
            left_hand = mid2arry(mid, list_tracks=[1])

            fr = filter(right_hand, nb_keep=1)
            fl = filter(left_hand, nb_keep=2)
            fd = combine([fr, fl])

            # Different folders
            right = 'right/'
            left = 'left/'
            dual = 'dual/'

            for e, head in enumerate([right, left, dual]):
                folder = 'head + name'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(head + name + '.npy', [fr, fl, fd][e])
            print(f"Time : {(time() - t0):.3f}")
