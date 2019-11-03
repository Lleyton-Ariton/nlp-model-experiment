import os
import parsing
import numpy as np
from keras.models import Input, Model
from keras.layers import Dense, Bidirectional, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping
from cortecx.construction import tools as t


class NounBaseline:

    def __init__(self, lstm_cells=128, shape=(60, 300), num_classes=23, epochs=150):
        self.lstm_cells = lstm_cells
        self.shape = shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.model = None
        self.callbacks = EarlyStopping(monitor='val_acc', patience=5)

    def build_model(self):
        input_layer = Input(shape=self.shape)
        x = Bidirectional(LSTM(self.lstm_cells, return_sequences=True))(input_layer)
        output_layer = TimeDistributed(Dense(self.num_classes, activation='softmax'))(x)

        model = Model(input_layer, output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    def train(self):
        data_set = parsing.load_conll('./data/CONLL2000/train.txt', task='conll-2000-noun')
        word_x_data = np.array(parsing.for_model_word_side(data_set))

        y_data = np.array(parsing.tag_output(data_set, t.noun_tags, num_classes=self.num_classes))

        data_set = parsing.load_conll('./data/CONLL2000/test.txt', task='conll-2000-noun')
        test_word_x_data = np.array(parsing.for_model_word_side(data_set))

        test_y_data = np.array(parsing.tag_output(data_set, t.noun_tags, num_classes=self.num_classes))

        self.model.fit(word_x_data, y_data, epochs=self.epochs, validation_data=(test_word_x_data, test_y_data),
                       callbacks=[self.callbacks])

    def save(self, fp: str):
        self.model.save(fp)


class POSBaseline:

    def __init__(self, lstm_cells=128, shape=(60, 300), num_classes=45, epochs=150):
        self.lstm_cells = lstm_cells
        self.shape = shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.model = None
        self.callbacks = EarlyStopping(monitor='val_acc', patience=5)

    def build_model(self):
        input_layer = Input(shape=self.shape)
        x = Bidirectional(LSTM(self.lstm_cells, return_sequences=True))(input_layer)
        output_layer = TimeDistributed(Dense(self.num_classes, activation='softmax'))(x)

        model = Model(input_layer, output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    def train(self):
        data_set = parsing.load_conll('./data/CONLL2000/train.txt', task='conll-2000-pos')
        word_x_data = np.array(parsing.for_model_word_side(data_set))

        y_data = np.array(parsing.tag_output(data_set, t.pos_pos_tags, num_classes=self.num_classes))

        data_set = parsing.load_conll('./data/CONLL2000/test.txt', task='conll-2000-pos')
        test_word_x_data = np.array(parsing.for_model_word_side(data_set))

        test_y_data = np.array(parsing.tag_output(data_set, t.pos_pos_tags, num_classes=self.num_classes))

        self.model.fit(word_x_data, y_data, epochs=self.epochs, validation_data=(test_word_x_data, test_y_data),
                       callbacks=[self.callbacks])

    def save(self, fp: str):
        self.model.save(fp)


class NERBaseline:
    def __init__(self, lstm_cells=128, shape=(60, 300), num_classes=10, epochs=150):
        self.lstm_cells = lstm_cells
        self.shape = shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.model = None
        self.callbacks = EarlyStopping(monitor='val_acc', patience=5)

    def build_model(self):
        input_layer = Input(shape=self.shape)
        x = Bidirectional(LSTM(self.lstm_cells, return_sequences=True))(input_layer)
        output_layer = TimeDistributed(Dense(self.num_classes, activation='softmax'))(x)

        model = Model(input_layer, output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    def train(self):
        data_set = parsing.load_conll('./data/CONLL2003/train.txt', task='conll-2003-ner')
        word_x_data = np.array(parsing.for_model_word_side(data_set))

        y_data = np.array(parsing.tag_output(data_set, t.ner_tags, num_classes=self.num_classes))

        data_set = parsing.load_conll('./data/CONLL2003/test.txt', task='conll-2003-ner')
        test_word_x_data = np.array(parsing.for_model_word_side(data_set))

        test_y_data = np.array(parsing.tag_output(data_set, t.ner_tags, num_classes=self.num_classes))

        self.model.fit(word_x_data, y_data, epochs=self.epochs, validation_data=(test_word_x_data, test_y_data),
                       callbacks=[self.callbacks])

    def save(self, fp: str):
        self.model.save(fp)
