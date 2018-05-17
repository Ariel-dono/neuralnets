import pickle
import gzip


def loading_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
    f.close()
    return train_set, valid_set, test_set



def save_state(toBeCached, layers, h_size):
    with open("/tmp/neural_net_state_" + str(layers) + "h_" + str(h_size), "wb") as fopened:
        pickle.dump(toBeCached, fopened)


def load_state(layers, h_size):
    with open("/tmp/neural_net_state_" + str(layers) + "h_" + str(h_size), "rb") as fopened:
        dict = pickle.load(fopened, encoding='bytes')
    return dict


def state_saved(layers, h_size):
    try:
        cache = open("/tmp/neural_net_state_" + str(layers) + "h_" + str(h_size), 'rb')
        cache.close()
        return True
    except FileNotFoundError:
        return False



def loading_data_xor():
    return [[[1, 0], [0, 1], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1], [1, 0],
             [0, 1], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [0, 1]],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]], \
           [[[0, 0], [1, 1], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1], [1, 0],
             [0, 1], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [0, 0]],
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]], []
