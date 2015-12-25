# coding: utf-8
import logging
logger = logging.getLogger("EntityExtractor_Model")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s:%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info("Started Logger")

import theano, keras
logger.info("Using Keras version %s" % keras.__version__)
logger.info("Using Theano version %s" % theano.__version__)
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import glob
import json
import time

import preprocess as pp

def vectorize_data(filenames, maxlen=100, output_label_size=6, output_label_dict=None):
    assert output_label_dict is not None
    X = []
    Y = []
    for i, filename in enumerate(filenames):
        for docid, doc in pp.get_documents(filename):
            for seq in pp.get_sequences(doc):
                x = []
                y = []
                for token in seq:
                    x.append(1 + token.word_index) # Add 1 to include token for padding
                    #y_vec = np.zeros(boundary_size)
                    y_idx = 1 + output_label_dict.get(token.b_label, -1) # Add 1 to include token for padding
                    #y_vec[y_idx] = 1
                    y.append(y_idx) # Add 1 to include token for padding
                X.append(x)
                Y.append(y)
    X = pad_sequences(X, maxlen=maxlen)
    Y = pad_sequences(Y, maxlen=maxlen)
    Y_vec = []
    # Now y should be converted to one hot vector for each index value
    for i in xrange(len(Y)):
        Y_vec.append([])
        for j in xrange(maxlen):
            #Y_vec[-1].append([])
            y_vec = np.zeros(output_label_size)
            y_vec[Y[i][j]] = 1
            Y_vec[-1].append(y_vec)
    X = np.array(X)
    Y = np.array(Y_vec)
    return X, Y

def gen_model(vocab_size=100, embedding_size=128, maxlen=100, output_size=6, hidden_layer_size=100, num_hidden_layers = 1, RNN_LAYER_TYPE="LSTM"):
    RNN_CLASS = LSTM
    if RNN_LAYER_TYPE == "GRU":
        RNN_CLASS = GRU
    logger.info("Parameters: vocab_size = %s, embedding_size = %s, maxlen = %s, output_size = %s, hidden_layer_size = %s, " %\
            (vocab_size, embedding_size, maxlen, output_size, hidden_layer_size))
    logger.info("Building Model")
    model = Sequential()
    logger.info("Init Model with vocab_size = %s, embedding_size = %s, maxlen = %s" % (vocab_size, embedding_size, maxlen))
    model.add(Embedding(vocab_size, embedding_size, input_length=maxlen))
    logger.info("Added Embedding Layer")
    model.add(Dropout(0.5))
    logger.info("Added Dropout Layer")
    for i in xrange(num_hidden_layers):
        model.add(RNN_CLASS(output_dim=hidden_layer_size, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
        logger.info("Added %s Layer" % RNN_LAYER_TYPE)
        model.add(Dropout(0.5))
        logger.info("Added Dropout Layer")
    model.add(RNN_CLASS(output_dim=output_size, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    logger.info("Added %s Layer" % RNN_LAYER_TYPE)
    model.add(Dropout(0.5))
    logger.info("Added Dropout Layer")
    model.add(TimeDistributedDense(output_size, activation="softmax"))
    logger.info("Added Dropout Layer")
    logger.info("Created model with following config:\n%s" % json.dumps(model.get_config(), indent=4))
    logger.info("Compiling model with optimizer %s" % optimizer)
    start_time = time.time()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f seconds." % total_time)
    return model


def gen_model_brnn(vocab_size=100, embedding_size=128, maxlen=100, output_size=6, hidden_layer_size=100, num_hidden_layers = 1, RNN_LAYER_TYPE="LSTM"):
    RNN_CLASS = LSTM
    if RNN_LAYER_TYPE == "GRU":
        RNN_CLASS = GRU
    logger.info("Parameters: vocab_size = %s, embedding_size = %s, maxlen = %s, output_size = %s, hidden_layer_size = %s, " %\
            (vocab_size, embedding_size, maxlen, output_size, hidden_layer_size))
    logger.info("Building Graph model for Bidirectional RNN")
    model = Graph()
    model.add_input(name='input', input_shape=(maxlen,), dtype=int)
    logger.info("Added Input node")
    logger.info("Init Model with vocab_size = %s, embedding_size = %s, maxlen = %s" % (vocab_size, embedding_size, maxlen))
    model.add_node(Embedding(vocab_size, embedding_size, input_length=maxlen), name='embedding', input='input')
    logger.info("Added Embedding node")
    model.add_node(Dropout(0.5), name="dropout_0", input="embedding")
    logger.info("Added Dropout Node")
    for i in xrange(num_hidden_layers):
        last_dropout_name = "dropout_%s" % i
        forward_name, backward_name, dropout_name = ["%s_%s" % (k, i + 1) for k in ["forward", "backward", "dropout"]]
        model.add_node(RNN_CLASS(output_dim=hidden_layer_size, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True), name=forward_name, input=last_dropout_name)
        logger.info("Added %s forward node[%s]" % (RNN_LAYER_TYPE, i+1))
        model.add_node(RNN_CLASS(output_dim=hidden_layer_size, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True, go_backwards=True), name=backward_name, input=last_dropout_name)
        logger.info("Added %s backward node[%s]" % (RNN_LAYER_TYPE, i+1))
        model.add_node(Dropout(0.5), name=dropout_name, inputs=[forward_name, backward_name])
        logger.info("Added Dropout node[%s]" % (i+1))
    model.add_node(TimeDistributedDense(output_size, activation="softmax"), name="tdd", input=dropout_name)
    logger.info("Added TimeDistributedDense node")
    model.add_output(name="output", input="tdd")
    logger.info("Added Output node")
    logger.info("Created model with following config:\n%s" % model.get_config())
    logger.info("Compiling model with optimizer %s" % optimizer)
    start_time = time.time()
    model.compile(optimizer, {"output": 'categorical_crossentropy'})
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f seconds." % total_time)
    return model



if __name__ == "__main__":
    CONFIG = json.load(open("config.json"))

    BASE_DATA_DIR = CONFIG["BASE_DATA_DIR"]
    DATA_DIR = "%s/%s" % (BASE_DATA_DIR, CONFIG["DATA_DIR"])
    vocab_file = "%s/%s" % (BASE_DATA_DIR, CONFIG["vocab_file"])
    boundary_file = "%s/%s" % (BASE_DATA_DIR, CONFIG["boundary_file"])
    category_file = "%s/%s" % (BASE_DATA_DIR, CONFIG["category_file"])
    BASE_OUT_DIR = CONFIG["BASE_OUT_DIR"]
    SAVE_MODEL_DIR = "%s/%s" % (BASE_OUT_DIR, CONFIG["SAVE_MODEL_DIR"])
    MODEL_PREFIX = CONFIG.get("MODEL_PREFIX", "model")
    maxlen = CONFIG["maxlen"]
    num_hidden_layers = CONFIG["num_hidden_layers"]
    embedding_size = CONFIG["embedding_size"]
    hidden_layer_size = CONFIG["hidden_layer_size"]
    RNN_LAYER_TYPE = CONFIG.get("RNN_LAYER_TYPE", "LSTM")
    optimizer = CONFIG["optimizer"]
    n_epochs = CONFIG["n_epochs"]
    save_every = CONFIG["save_every"]
    model_type = CONFIG.get("model_type", "rnn") # rnn, brnn

    RNN_CLASS = LSTM
    if RNN_LAYER_TYPE == "GRU":
        RNN_CLASS = GRU


    index_word, word_dict = pp.load_vocab(vocab_file)
    pp.WordToken.set_vocab(word_dict = word_dict)
    index_boundary, boundary_dict = pp.load_vocab(boundary_file)
    index_category, category_dict = pp.load_vocab(category_file)
    vocab_size = len(index_word) + pp.WordToken.VOCAB + 1 # Add offset of VOCAB and then extra token for padding
    boundary_size = len(index_boundary) + 1 # Add extra token for padding 
    category_size = len(index_category) + 1 # Add extra token for padding

    logger.info("Parameters: vocab_size = %s, embedding_size = %s, maxlen = %s, boundary_size = %s, category_size = %s, embedding_size = %s, hidden_layer_size = %s" %\
                    (vocab_size, embedding_size, maxlen, boundary_size, category_size, embedding_size, hidden_layer_size))

    # Read the data
    CV_filenames = [glob.glob("%s/%s/*.xml" % (DATA_DIR, i)) for i in range(1,6)]

    train_files = reduce(lambda x, y: x + y, CV_filenames[0:1])
    test_files = reduce(lambda x, y: x + y, CV_filenames[0:1])

    X_train, Y_train = vectorize_data(train_files, maxlen=maxlen, output_label_size=boundary_size, output_label_dict=boundary_dict)
    X_test, Y_test = vectorize_data(test_files, maxlen=maxlen, output_label_size=boundary_size, output_label_dict=boundary_dict)

    logger.info("Loaded data shapes:\nX_train: %s, Y_train: %s\nX_test: %s, Y_test: %s" % (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))

    if model_type == "brnn":
        model = gen_model_brnn(vocab_size=vocab_size, embedding_size=embedding_size, maxlen=maxlen, output_size=boundary_size, hidden_layer_size=hidden_layer_size, num_hidden_layers = num_hidden_layers, RNN_LAYER_TYPE=RNN_LAYER_TYPE)
    else:
        model = gen_model(vocab_size=vocab_size, embedding_size=embedding_size, maxlen=maxlen, output_size=boundary_size, hidden_layer_size=hidden_layer_size, num_hidden_layers = num_hidden_layers, RNN_LAYER_TYPE=RNN_LAYER_TYPE)


    for epoch in range(0, n_epochs, save_every):
        logger.info("Starting Epochs %s to %s" % (epoch, epoch + save_every))
        start_time = time.time()
        if model_type == "brnn":
            model.fit({"input": X_train,"output": Y_train}, validation_data={"input": X_test, "output": Y_test}, nb_epoch=save_every, verbose=2)
        else:
            model.fit(X_train,Y_train, validation_data=(X_test, Y_test), nb_epoch=save_every, verbose=2, show_accuracy=True)
        total_time = time.time() - start_time
        logger.info("Finished training %.3f epochs in %s seconds with %.5f seconds/epoch" % (save_every, total_time, total_time * 1.0/ save_every))
        model.save_weights("%s/%s_%s.h5" % (SAVE_MODEL_DIR, MODEL_PREFIX, epoch), overwrite=True)
