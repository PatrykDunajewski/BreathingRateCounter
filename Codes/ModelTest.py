import os 
import numpy as np
import pandas as pd
import librosa
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Conv1D, MaxPooling1D, Input


CHUNK = 4096
N_MFCC = 40
BATCH_SIZE = 4
EPOCHS = 100
EPOCHS_2 = 30
CALLBACK = tf.keras.callbacks.EarlyStopping(monitor = 'loss', min_delta = 1e-5, patience = 10, restore_best_weights = True)

def data_converter(record_type, train_test):
    tmp_array = np.empty((0,2))
    basic_path = f'./Clips/{train_test}/{record_type}'
    for label_type in os.listdir(basic_path):
        for file_path in os.listdir(f'{basic_path}/{label_type}'):
            
            if str(label_type).lower() == "inhale":
                label_id = 0
            else:
                label_id = 1

            audio, sample_rate = librosa.load(f'{basic_path}/{label_type}/{file_path}')

            for i in range(math.floor(len(audio)/CHUNK)):
                segments = audio[i*CHUNK:(i+1)*CHUNK]
                mfccs_features = librosa.feature.mfcc(y=segments, sr=sample_rate, n_mfcc=N_MFCC)
                mfccs_scaled_features = np.mean(mfccs_features.T, axis=0).reshape(N_MFCC)
                tmp_array = np.append(tmp_array, np.array([[mfccs_scaled_features, label_id]]),axis=0)

    return(pd.DataFrame(tmp_array, columns = ["data", "label"]))

def create_models(act_function, X_train, y_train):

    
    #model res
    inputs = Input(shape = (40,))
    dense1 = Dense(100, input_shape=(40,), activation = act_function)(inputs)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(100)(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    sum1 = tf.keras.layers.add([dropout1, dropout2])
    outputs = Dense(1, activation = 'sigmoid')(sum1)
    modelRes = Model(inputs, outputs)
    modelRes.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    modelRes.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[CALLBACK])
    
    #normal model
    model=Sequential()
    model.add(Dense(100, input_shape=(40,), activation = act_function))
    model.add(Dropout(0.2))
    model.add(Dense(200))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[CALLBACK])
    
    #rnn model
    modelRNN = Sequential()
    modelRNN.add(LSTM(128, input_shape = (40,1), return_sequences=False))
    modelRNN.add(Dense(1, activation="sigmoid"))
    modelRNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
    modelRNN.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[CALLBACK])

    #CNN model
    modelCNN = Sequential()
    modelCNN.add(Conv1D(filters=64, kernel_size=3, activation=act_function, input_shape=(40,1)))
    modelCNN.add(Dropout(0.2))
    modelCNN.add(MaxPooling1D(pool_size=2))
    modelCNN.add(Flatten())
    modelCNN.add(Dense(100, activation=act_function))
    modelCNN.add(Dense(1, activation="sigmoid"))
    modelCNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    modelCNN.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[CALLBACK])

    return [[model, "NN"], [modelRes, "Res"], [modelCNN, "CNN"], [modelRNN, "RNN"]]

def model_fit(X_train, y_train, model):
    model.fit(X_train, y_train, batch_size=1, epochs=EPOCHS_2, callbacks=[CALLBACK])
    return model

def model_learning(basic_models, recording_type):
    new_models = []
    basic_models_copy = []

    for model, model_name in basic_models:
        new_model = clone_model(model)
        new_model.set_weights(model.get_weights())
        new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        basic_models_copy.append([new_model, model_name])

    train_df = data_converter(recording_type, "train")
    X_train = np.array(train_df["data"].tolist())
    y_train = np.array(train_df["label"].tolist())
    for model, model_name in basic_models_copy:
        new_models.append([model_fit(X_train, y_train, model), model_name])
    return [new_models, recording_type]
        

def main():
    recording_types_array = [] 
    for records_type in os.listdir("./Clips/train"):
        recording_types_array.append(records_type)
    basic_train_df = data_converter(recording_types_array[0], "train")
    X_train = np.array(basic_train_df["data"].tolist())
    y_train = np.array(basic_train_df["label"].tolist())
    
    basic_models = create_models("sigmoid", X_train, y_train)
    new_models = []
    for recording_type in recording_types_array[1:]:
        new_models.append(model_learning(basic_models, recording_type))


    # model, modelRes, modelCNN, modelRNN = create_models("sigmoid", X_train, y_train)
    # basic_models = [model, modelRes, modelCNN, modelRNN]

    # model_relu, modelRes_relu, modelCNN_relu, modelRNN_relu = create_models("relu", X_train, y_train)
    # ,[model_relu, "NN", "relu"], [modelRNN_relu, "RNN", "relu"], [modelCNN_relu, "CNN", "relu"], [modelRes_relu, "Res", "relu"]


    test_results = []
    for record_type in recording_types_array:
        test_df = data_converter(record_type, "test")
        X_test = np.array(test_df["data"].tolist())
        y_test = np.array(test_df["label"].tolist())
        for model_type, model_name in basic_models:
            test_acc = model_type.evaluate(X_test, y_test)
            test_results.append([model_name, record_type, 0, float(round(test_acc[1],4))])
        
        for new_model in new_models:
            if record_type == new_model[1]:
                for recording_type_model, model_name in new_model[0]:
                    test_acc = recording_type_model.evaluate(X_test, y_test)
                    test_results.append([model_name, record_type, 1, float(round(test_acc[1],4))])

    results_df = pd.DataFrame(test_results, columns = ["model_name", "record_type", "trained_again", "acc"])
    results_df.to_csv("./Results2.csv", header = True, index = False, sep=";")

if __name__ == '__main__':
    main()

