import streamlit as st
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

from glob import glob

from music21 import instrument, note, chord, tempo, duration, stream

from PIL import Image

from streamlit.report_thread import get_report_ctx

from logzero import logger, logfile

import sys

def exit_on_exception(e, func_string):
    error_class = str(e.__class__).split()[1]
    message = '{} on {}: {}'.format(error_class[1:-2], func_string, str(e))
    logger.exception(message)
    sys.exit(message)

@st.cache()
def init_logfile(session_id):
    """Initialize logfile for the session_id. Caching this function assures that
    the initialization only happens once for each session."""
    
    logfile_name = './logfiles/logzero_{}.log'.format(session_id)
    logfile(logfile_name)
    logger.info('Logfile {} created'.format(logfile_name))

# Following hash_funcs arg was needed to resolve an error:
import tensorflow.python.keras.engine as e
@st.cache(hash_funcs={e.sequential.Sequential: id})
def lstm(n_lstm_layers = 2, n_dense_layers = 1, n_lstm_nodes = 512, \
         dropout_rate = 0.4, n_keys_piano = 88, window_size = 16, \
         n_dur_nodes = 20):
    """Generate a keras Sequential model of the form as described in Figure 16
    of https://www.tandfonline.com/doi/full/10.1080/25765299.2019.1649972"""
    
    model = Sequential()
    model.add(LSTM(n_lstm_nodes, return_sequences = True, input_shape = \
                   (window_size, n_keys_piano + n_dur_nodes,)))
    model.add(Dropout(dropout_rate))
    for i in range(1, n_lstm_layers - 1):
        model.add(LSTM(n_lstm_nodes, return_sequences = True))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(n_lstm_nodes))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_lstm_nodes // 2))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    for i in range(n_dense_layers - 1):
        model.add(Dense(n_lstm_nodes // 2))
        model.add(Dropout(dropout_rate))
    model.add(Dense(n_keys_piano + n_dur_nodes))
    model.add(Activation('sigmoid'))
    return model

# Same hash_funcs as above
@st.cache(hash_funcs={e.sequential.Sequential: id})
def get_weights(model, filepath):
    """Load the model weights onto the architecture skeleton model. This method
    saves on storage space and is sufficient for performing inference."""

    try:
        model.load_weights(filepath)
    except (OSError, ValueError) as e:
        exit_on_exception(e, 'model.load_weights() in get_weights()')
    summary = []
    
    try:
        model.summary(print_fn = lambda x: summary.append(x))
    except ValueError as e:
        exit_on_exception(e, 'model.summary() in get_weights()')
    return model, '\n'.join(summary)

@st.cache()
def load_validation_set(filepath, max_duration = 6.545454545454545):
    """Load the X part of the validation set (from multiple files), and reverse
    the duration scaling applied previously so that the duration is in
    seconds. We take the max_duration as input because to calculate it would
    require loading the training set and the y part of the validation set."""

    files = glob(filepath + '_*.npy')
    try:
        X_val = np.load(files[0])
        for f in files[1:]:
            temp = np.load(f)
            try:
                X_val = np.append(X_val, temp, axis = 0)
            except ValueError as e:
                exit_on_exception(e, 'np.append() in load_validation_set()')
    except (IOError, ValueError) as e:
        exit_on_exception(e, 'np.load() in load_validation_set()')
        
    X_val[:, :, -1] *= max_duration    # convert back into seconds
    return X_val

@st.cache()
def init_rndm_seed_index_file(filepath, index = 42):
    """Just initialize the file with index. Since it has the st.cache()
    decorator, it will only be called if the inputs change, thus just once."""

    with open(filepath, 'w') as f:
        f.write(str(index))

@st.cache(hash_funcs={e.sequential.Sequential: id}, \
                      allow_output_mutation = True)
# to suppress a warning when no_of_timesteps or threshold are changed                    
def generate_musical_sequence(model, random_music, n_keys_piano = 88, \
                              n_dur_nodes = 20, no_of_timesteps = 16,
                              threshold = 0.5):
    """Generate a musical sequence using model and from the sequence
    random_music as a starting point. We will perform an inference
    (and evaluation using threshold) no_of_timesteps times, each time
    appending the prediction onto random_music. After each prediction,
    the oldest vector in the sequence is popped until all of the
    original vectors in random_music are gone. We still keep adding
    on to random_music but we perform inference on the last 16 (as
    this was the window_size used during training). Finally, return
    both the original random_music and the new one."""

    window_size = random_music.shape[0]
    original_random_music = random_music.copy()   # shallow copy
    for i in range(no_of_timesteps):
        try:
            reshaped = random_music.reshape(1, random_music.shape[0], \
                                               random_music.shape[1])
        except ValueError as e:
            exit_on_exception(e, 'ndarray.reshape() in ' + \
                              'generate_musical_sequence()')
        try:
            prob = model.predict(reshaped[:, -window_size:])[0]
        except ValueError as e:
            exit_on_exception(e, 'model.predict() in ' + \
                              'generate_musical_sequence()')
            
        y_pred = [0 if p < threshold else 1 for p in prob[:-n_dur_nodes]] + \
                 list(np.tile([np.mean(prob[-n_dur_nodes:])], n_dur_nodes))

        try:
            if (i >= window_size):
                random_music = np.insert(random_music, len(random_music), \
                                         y_pred, axis = 0)
            else:
                random_music = np.insert(random_music, len(random_music), \
                                         y_pred, axis = 0)[1:]
        except ValueError as e:
            exit_on_exception(e, 'np.insert() in ' + \
                              'generate_musical_sequence()')
            
    # Need just one duration now that we're done with inference
    return original_random_music[:, :-n_dur_nodes + 1], \
           random_music[:, :-n_dur_nodes + 1]       

@st.cache()
def transpose_sequence(sequence, transposition = 0):
    """Perform a left-shift on the keys' part of the vectors
      Effectively, this outputs a new sequence repesenting
      a song but transposed. The size of the shift is
      transposition"""
    if (transposition == 0):
        return sequence
    shift = transposition
    sequence, durations = sequence[:, :-1], sequence[:, -1]
    
    try:
        for i in range(len(sequence)):
            sequence[i] = np.concatenate((sequence[i][shift:], \
                                          sequence[i][:shift]))
    except ValueError as e:
        exit_on_exception(e, 'np.concatenate() in ' + 'transpose_sequence()')

    try:
        return np.insert(sequence, len(sequence[0]), durations, axis = 1)
    except (IndexError, ValueError) as e:
        exit_on_exception(e, 'np.insert() in ' + 'transpose_sequence()')


@st.cache()
def piano_idx_to_note(index):
    """Return the note corresponding to the input, index"""
    
    all_notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', \
                 'G', 'G#']
    note = all_notes[index % len(all_notes)]
    octave = (index + 9) // len(all_notes)
    return note + str(octave)

@st.cache()
def convert_to_midi(sequence, bpm = 60, output_file = './midi_output/music.mid'):
    """Save sequence as a midi file (with path = output_file). sequence
    can be from the original dataset or a new sequence generated by a
    trained model"""
    offset = 0    # the distance in quarter-notes of the note/chord/rest
                  # being written from the beginning
    output_notes = [instrument.Piano(), tempo.MetronomeMark(number = bpm)]

    bps = bpm / 60    # beats per second
    converted_duration = duration.Duration()
    # create note, chord, and rest objects
    for vector in sequence:
        # convert from seconds to beats
        converted_duration.quarterLength = vector[-1] * bps
        
        if (np.sum(vector[:-1]) > 1):      # chord
            indices_in_chord = np.argsort(vector[:-1])[-int(np.sum(\
                vector[:-1])):]
            notes_in_chord = [piano_idx_to_note(i) for i in indices_in_chord]
            notes = []
            for cur_note in notes_in_chord:
                new_note = note.Note(cur_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.duration = converted_duration
            output_notes.append(new_chord)
            
        elif (np.sum(vector[:-1]) == 1):   # note
            index = np.argmax(vector[:-1])
            new_note = piano_idx_to_note(index)
            new_note = note.Note(new_note)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration = converted_duration
            output_notes.append(new_note)
        
        elif (np.sum(vector[:-1]) == 0):   # rest
            new_rest = note.Rest()
            new_rest.offset = offset
            new_rest.duration = converted_duration
            output_notes.append(new_rest)
                                                               
        offset += vector[-1]
                                                               
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp = output_file)

from os import path
from base64 import b64encode
def get_binary_file_downloader_html(bin_file, file_label='File'):

    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
    except FileNotFoundError as e:
        exit_on_exception(e, 'open() in ' + 'get_binary_file_downloader_html()')
        
    bin_str = b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{path.basename(bin_file)}">Download {file_label}</a>'
    return href
    
if __name__ == '__main__':

    session_id = get_report_ctx().session_id  # Get Session ID
    init_logfile(session_id)
    
    st.title('ChopinBot 1.0: Music Generation with an Long Short-Term' \
             + ' Memory (LSTM) Neural Network')

    model = lstm(n_lstm_nodes = 1024)
    model, summary = get_weights(model, './models/best_maestro_model_weights'+\
                                 '_ext20_2_1_1024_0pt4_mnv_2.h5')
             
    X_val = load_validation_set('./X_val/X_val_ext')

    rndm_seed_index_path = './rndm_seed_index_files/rndm_seed_index_{}.txt'.\
                           format(session_id)
    init_rndm_seed_index_file(rndm_seed_index_path)

    portrait = Image.open('./images/chopin.jpg')

    caption = '3D Portrait of Fr\u00E9d\u00E9ric Chopin from ' + \
              '[here](https://hadikarimi.com/portfolio/frederic-chopin)'

    st.image(portrait, use_column_width = True)
    tabs = '&emsp;' * 48
    st.markdown(tabs + caption.encode('utf-8').decode('utf-8'))

    precision_and_recall = Image.open('./images/precision_and_recall.jpg')
    caption = 'The [precision and recall](https://en.wikipedia.org/wiki/Preci'+\
              'sion_and_recall) can be adjusted by changing the probability t'+\
              'hreshold. Lowering the threshold will increase the number of n'+\
              'otes but comes at the cost of predicting more "wrong" notes. T'+\
              'he f-measure is the [harmonic mean](https://en.wikipedia.org/w'+\
              'iki/Harmonic_mean) of the two.'
    st.image(precision_and_recall, use_column_width = True)
    st.markdown(caption)
    
    st.sidebar.title('Controls:')

    if (st.sidebar.checkbox('Manually Set Seed Index')):
        seed_index = st.sidebar.number_input('Set Seed Index in range [0, {}]'\
                     .format(len(X_val) - 1), min_value = 0, \
                     max_value = len(X_val) - 1, value = 42, key = 'seed')
    else:     # Set Set Index Randomly
        if (st.sidebar.button('Generate New Seed Index')):
            seed_index = np.random.randint(len(X_val))
            with open(rndm_seed_index_path, 'w') as f:
                f.write(str(seed_index))
        else: # Read from file
            with open(rndm_seed_index_path, 'r') as f:
                seed_index = int(f.read())
        st.sidebar.write('Seed Index = {}'.format(seed_index))
  #  st.sidebar.write('Seed Index = {}'.format(seed_index))

    no_of_timesteps = st.sidebar.slider('# of timesteps to predict', 1, \
                                        320, 16)

    st.sidebar.write('no_of_timesteps = {}'.format(no_of_timesteps))
    
    threshold = st.sidebar.number_input(\
        'Set probability threshold in range (0, 1)', min_value = 0.0, \
        max_value = 1.0, value = 0.5, key = 'threshold')

  #  st.sidebar.write('threshold = {}'.format(threshold))

    seed_music, generated_music = generate_musical_sequence(model, \
                                  X_val[seed_index], no_of_timesteps = \
                                  no_of_timesteps, threshold = threshold)
    
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = st.sidebar.selectbox('Choose a musical key', keys)

 #   st.sidebar.write('key = {}'.format(key))

    if (key == 'C'):
        transposition = 0
    else:
        transposition = 12 - keys.index(key)  # transpose leftward
    transposed_generated_music = transpose_sequence(generated_music, \
                                        transposition = transposition)
    transposed_seed_music = transpose_sequence(seed_music, transposition = \
                                           transposition)

    bpm = st.sidebar.number_input('Set the bpm (beats per minute) in the '\
                        'range [20, 180]', min_value = 20, max_value = 180,\
                         value = 60, key = 'bpm')

 #  st.sidebar.write('bpm = {}'.format(bpm))

    for music_type in ['seed', 'generated']:                                                             
        if (st.button('Create MIDI File for the ' + music_type.title() + \
                      ' Music', key = music_type)):
            exec(('convert_to_midi(transposed_{0}_music, bpm = bpm, ' + \
                 'output_file = \'/tmp/{0}.mid\')').format(music_type))
            st.markdown(get_binary_file_downloader_html('/tmp/' + \
                    music_type + '.mid', 'MIDI'), unsafe_allow_html = True)


    

