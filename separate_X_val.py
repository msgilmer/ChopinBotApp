from glob import glob
import numpy as np
from sys import getsizeof

"""This script is designed to convert the X part of the validation set to a 
more space-efficient format for loading and caching in ChopinBotApp.py. First,
I keep only one of the last n_dur_nodes elements of each vector since they each
hold the exact same value. Second, since most of the vector can be saved as a
np.bool type, I do this and save a separate numpy array for the durations."""


def load_validation_set(filepath, max_duration = 6.545454545454545):
    """Load the X part of the validation set (from multiple files), and reverse
    the duration scaling applied previously so that the duration is in
    seconds. We take the max_duration as input because to calculate it would
    require loading the training set and the y part of the validation set."""

    files = glob(filepath + '_*.npy')
    X_val = np.load(files[0])
    for f in files[1:]:
        temp = np.load(f)
        X_val = np.append(X_val, temp, axis = 0)
        
    X_val[:, :, -1] *= max_duration    # convert back into seconds
    return X_val
            
if __name__ == '__main__':

    X_val = load_validation_set('./X_val/X_val_ext')
    print('X_val: ', getsizeof(X_val) / (2**20), 'MB')

    n_dur_nodes = 20    # The number of elements containing the identical
                        # duration in the validation set.

    sequences, durations = X_val[:, :, :-n_dur_nodes], X_val[:, :, -1]

    booleanized_sequences = sequences.astype(np.bool)

    print('booleanized_sequences: ', getsizeof(booleanized_sequences) / (2**20)\
                                                                        , 'MB')
    print('durations: ', getsizeof(durations) / (2**20), 'MB')

    original_sequences = booleanized_sequences.astype(np.float64)

    # Assert nothing is lost between the conversion to np.bool and back to
    # np.float64
    assert(np.array_equal(sequences, original_sequences))

    np.save('X_val/booleanized_sequences.npy', booleanized_sequences)
    np.save('X_val/durations.npy', durations)

    # Test out the method for converting back to original format from the
    # energy-efficient format saved above
    seed_index = 42
    random_music = np.concatenate([booleanized_sequences[seed_index], \
                    np.repeat(np.expand_dims(durations[seed_index], -1), \
                                     n_dur_nodes, axis = 1)], axis = 1)

    print('random_music.shape = ', random_music.shape)
    print('random_music.dtype = ', random_music.dtype)
    
