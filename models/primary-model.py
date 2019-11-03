import os
import parsing
import numpy as np
from keras.models import Input, Model
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense, concatenate
from keras.callbacks import EarlyStopping
from keras_contrib.layers import CRF
from keras_contrib import losses
from cortecx.construction import tools as t


class NounModel:

    def __init__(self):
        self.model = None

    def build_model(self):
        word_in_layer = Input(shape=(60, 300))
        char_in_layer = Input(shape=(60, 10, 300))

        char_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, recurrent_dropout=0.2)))(
            char_in_layer)

        tree = concatenate([word_in_layer, char_branch])
        tree = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2))(tree)
        tree = TimeDistributed(Dense(23, activation='relu'))(tree)

        crf = CRF(23)

        output_layer = crf(tree)

        model = Model([word_in_layer, char_in_layer], output_layer)
        model.compile(optimizer='adam', loss=losses.crf_loss, metrics=['accuracy'])

        self.model = model

    def train(self):
        data_set = parsing.load_conll('./data/CONLL2000/train.txt', task='conll-2000-noun')

        word_x_data = parsing.for_model_word_side(data_set)
        char_x_data = parsing.for_model_char_side(data_set)

        y_data = np.array(parsing.tag_output(data_set, t.noun_tags, num_classes=23))

        data_set = parsing.load_conll('./data/CONLL2000/test.txt', task='conll-2000-noun')

        test_word_x_data = parsing.for_model_word_side(data_set)
        test_char_x_data = parsing.for_model_char_side(data_set)

        test_y_data = np.array(parsing.tag_output(data_set, t.noun_tags, num_classes=23))

        callbacks = EarlyStopping(monitor='val_acc', patience=5)
        self.model.fit([word_x_data, char_x_data], y_data, epochs=150,
                       validation_data=([test_word_x_data, test_char_x_data], test_y_data), callbacks=[callbacks])

    def save(self, fp: str):
        self.model.save(fp)


class POSModel:

    def __init__(self):
        self.model = None

    def build_model(self):
        word_in_layer = Input(shape=(60, 300))
        char_in_layer = Input(shape=(60, 10, 300))

        char_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, recurrent_dropout=0.2)))(
            char_in_layer)

        tree = concatenate([word_in_layer, char_branch])
        tree = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2))(tree)
        tree = TimeDistributed(Dense(45, activation='relu'))(tree)

        crf = CRF(45)

        output_layer = crf(tree)

        model = Model([word_in_layer, char_in_layer], output_layer)
        model.compile(optimizer='adam', loss=losses.crf_loss, metrics=['accuracy'])

        self.model = model

    def train(self):
        data_set = parsing.load_conll('./data/CONLL2000/train.txt', task='conll-2000-pos')

        word_x_data = parsing.for_model_word_side(data_set)
        char_x_data = parsing.for_model_char_side(data_set)

        y_data = np.array(parsing.tag_output(data_set, t.pos_pos_tags))

        data_set = parsing.load_conll('./data/CONLL2000/test.txt', task='conll-2000-pos')

        test_word_x_data = parsing.for_model_word_side(data_set)
        test_char_x_data = parsing.for_model_char_side(data_set)

        test_y_data = np.array(parsing.tag_output(data_set, t.pos_pos_tags))

        callbacks = EarlyStopping(monitor='val_acc', patience=5)
        self.model.fit([word_x_data, char_x_data], y_data, epochs=150,
                       validation_data=([test_word_x_data, test_char_x_data], test_y_data), callbacks=[callbacks])

    def save(self, fp: str):
        self.model.save(fp)


class NERModel:

    def __init__(self):
        self.model = None

    def build_model(self):
        word_in_layer = Input(shape=(60, 300))
        char_in_layer = Input(shape=(60, 10, 300))

        char_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, recurrent_dropout=0.2)))(
            char_in_layer)

        tree = concatenate([word_in_layer, char_branch])
        tree = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2))(tree)
        tree = TimeDistributed(Dense(10, activation='relu'))(tree)

        crf = CRF(10)

        output_layer = crf(tree)

        model = Model([word_in_layer, char_in_layer], output_layer)
        model.compile(optimizer='adam', loss=losses.crf_loss, metrics=['accuracy'])

        self.model = model

    def train(self):
        data_set = parsing.load_conll('./data/CONLL2000/train.txt', task='conll-2003-ner')

        word_x_data = parsing.for_model_word_side(data_set)
        char_x_data = parsing.for_model_char_side(data_set)

        y_data = np.array(parsing.tag_output(data_set, t.ner_tags, num_classes=10))

        data_set = parsing.load_conll('./data/CONLL2000/test.txt', task='conll-2003-ner')

        test_word_x_data = parsing.for_model_word_side(data_set)
        test_char_x_data = parsing.for_model_char_side(data_set)

        test_y_data = np.array(parsing.tag_output(data_set, t.ner_tags, num_classes=10))

        callbacks = EarlyStopping(monitor='val_acc', patience=5)
        self.model.fit([word_x_data, char_x_data], y_data, epochs=150,
                       validation_data=([test_word_x_data, test_char_x_data], test_y_data), callbacks=[callbacks])

    def save(self, fp: str):
        self.model.save(fp)
