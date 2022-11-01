import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
import matplotlib.pyplot as plt


CHUNK_BS = 2048 * 32

RATE = 44100
CHUNK_ARRAY = [2048, 4096]
N_MFCC_ARRAY = [20, 40]
BATCH_SIZE = 16
EPOCHS = 100
CALLBACK = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 1e-5, patience = 10, restore_best_weights = True)
CHUNK_X_ARRAY = [10, 20]



def approx(val):
    if val >=0.5:
        return math.ceil(val)
    else:
        return math.floor(val)


def create_vals(dict_v, path, chunk):
    for afile in os.listdir(path):
        if afile.split('-')[0] not in dict_v:
            dict_v[afile.split('-')[0]] = {}
            dict_v[afile.split('-')[0]]['labels'] = []
            dict_v[afile.split('-')[0]]['mean_abs_val'] = []
            dict_v[afile.split('-')[0]]['n_breaths'] = 0
            for mfcc_num in N_MFCC_ARRAY:
                dict_v[afile.split('-')[0]][f'mfcc_{mfcc_num}'] = []

        audio, sample_rate = librosa.load(f'{path}/{afile}',sr = RATE)
        dict_v[afile.split('-')[0]]['n_breaths'] += 1
        if int(afile.split('-')[1].split('.')[0])%2==0:
            label_id = 1
        else:
            label_id = 0
        
        for i in range(math.floor(len(audio)/chunk)):
            segments = audio[i*chunk:(i+1)*chunk]
            for mfcc_num in N_MFCC_ARRAY:
                mfccs_features = librosa.feature.mfcc(y=segments, sr=sample_rate, n_mfcc=mfcc_num)
                mfcc_val = np.mean(mfccs_features.T, axis=0).reshape(mfcc_num)
                dict_v[afile.split('-')[0]][f'mfcc_{mfcc_num}'].append(mfcc_val)
            dict_v[afile.split('-')[0]]['mean_abs_val'].append(np.mean(np.abs(segments)))
            dict_v[afile.split('-')[0]]['labels'].append(label_id)

def modelRNN(X_train, y_train):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(128))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
    model.fit(X_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[CALLBACK])
    return model

def modelBiRNN(X_train, y_train):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
    model.fit(X_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[CALLBACK])
    return model

def modelNN(X_train, y_train):
    model=Sequential()
    model.add(Dense(100, input_shape=(X_train.shape[1],), activation = 'sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(200))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
    model.fit(X_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[CALLBACK]) 
    return model

def dict_results_creator(dict_res_tmp, dict_key):
    dict_res_tmp[dict_key] = {}
    dict_res_tmp[dict_key]['n_preds'] = 0
    dict_res_tmp[dict_key]['correct_preds'] = 0
    dict_res_tmp[dict_key]['n_breaths'] = 0
    dict_res_tmp[dict_key]['correct_breaths'] = 0
    return dict_res_tmp[dict_key]

def create_train_test(dict_train, dict_test, dict_res, chunk):
    for mfcc_num in N_MFCC_ARRAY:
        X_train_NN = []
        y_train = []

        for dict_keys in list(dict_train.keys()):
            X_train_NN.extend(dict_train[dict_keys][f'mfcc_{mfcc_num}'])
            y_train.extend(dict_train[dict_keys]['labels'])

        model_NN = modelNN(np.array(X_train_NN), np.array(y_train))

        dict_res[f'NN_{mfcc_num}_{chunk}'] = {}
        for dict_keys in list(dict_test.keys()):
            prev_pred = -1
            dict_res_tmp = dict_results_creator(dict_res[f'NN_{mfcc_num}_{chunk}'], dict_keys)
            dict_res_tmp['correct_breaths'] = dict_test[dict_keys]['n_breaths']

            for idx, val in enumerate(dict_test[dict_keys][f'mfcc_{mfcc_num}']):
                dict_res_tmp['n_preds'] += 1
                pred = approx(model_NN.predict(np.array([val])))
                if pred == dict_test[dict_keys]['labels'][idx]:
                    dict_res_tmp['correct_preds'] += 1
                if pred!=prev_pred:
                    dict_res_tmp['n_breaths'] += 1
                prev_pred = pred
        
        # df = pd.DataFrame(dict_res)
        # print(df)
        # df.to_csv('./Final_result.csv')

        for chunk_x in CHUNK_X_ARRAY:
            X_train_RNN = []
            RNN_array = [[*[-1000]*mfcc_num, -1]]* (chunk_x - 1)
            for dict_keys in list(dict_train.keys()):
                X_train_RNN.append(np.array([*RNN_array, [*dict_train[dict_keys][f'mfcc_{mfcc_num}'][0], dict_train[dict_keys]['labels'][0]]]))
                for i in range(1,np.array(dict_train[dict_keys][f'mfcc_{mfcc_num}']).shape[0]):
                    X_train_RNN.append(np.array([*X_train_RNN[-1][-chunk_x+1:], [*dict_train[dict_keys][f'mfcc_{mfcc_num}'][i], dict_train[dict_keys]['labels'][i]]]))       
                    
            for val in X_train_RNN:
                val[-1][-1] = -1
            
            model_RNN = modelRNN(np.array(X_train_RNN), np.array(y_train))

            dict_res[f'RNN_{mfcc_num}_{chunk}_{chunk_x}'] = {}
            for dict_keys in list(dict_test.keys()):
                val_array = [[*[-1000]*mfcc_num, -1]]* (chunk_x)
                prev_pred = -1
                dict_res_tmp = dict_results_creator(dict_res[f'RNN_{mfcc_num}_{chunk}_{chunk_x}'], dict_keys)
                dict_res_tmp['correct_breaths'] = dict_test[dict_keys]['n_breaths']

                for idx, val in enumerate(dict_test[dict_keys][f'mfcc_{mfcc_num}']):
                    dict_res_tmp['n_preds'] += 1
                    val_array = [*val_array[1:], [*val, -1]] 
                    pred = approx(model_RNN.predict(np.array([val_array])))
                    if pred == dict_test[dict_keys]['labels'][idx]:
                        dict_res_tmp['correct_preds'] += 1
                    if pred!=prev_pred:
                        dict_res_tmp['n_breaths'] += 1
                    prev_pred = pred
                    val_array[-1][-1] = pred

            model_BiRNN = modelBiRNN(np.array(X_train_RNN), np.array(y_train))

            dict_res[f'BiRNN_{mfcc_num}_{chunk}_{chunk_x}'] = {}
            for dict_keys in list(dict_test.keys()):
                val_array = [[*[-1000]*mfcc_num, -1]]* (chunk_x)
                prev_pred = -1
                dict_res_tmp = dict_results_creator(dict_res[f'BiRNN_{mfcc_num}_{chunk}_{chunk_x}'], dict_keys)
                dict_res_tmp['correct_breaths'] = dict_test[dict_keys]['n_breaths']

                for idx, val in enumerate(dict_test[dict_keys][f'mfcc_{mfcc_num}']):
                    dict_res_tmp['n_preds'] += 1
                    val_array = [*val_array[1:], [*val, -1]] 
                    pred = approx(model_BiRNN.predict(np.array([val_array])))
                    if pred == dict_test[dict_keys]['labels'][idx]:
                        dict_res_tmp['correct_preds'] += 1
                    if pred!=prev_pred:
                        dict_res_tmp['n_breaths'] += 1
                    prev_pred = pred
                    val_array[-1][-1] = pred


            X_train_RNN = []
            RNN_array = [[*[-1000]*mfcc_num, -1000, -1]]* (chunk_x - 1)
            for dict_keys in list(dict_train.keys()):
                X_train_RNN.append(np.array([*RNN_array, [*dict_train[dict_keys][f'mfcc_{mfcc_num}'][0], dict_train[dict_keys]['mean_abs_val'][0], dict_train[dict_keys]['labels'][0]]]))
                for i in range(1,np.array(dict_train[dict_keys][f'mfcc_{mfcc_num}']).shape[0]):
                    X_train_RNN.append(np.array([*X_train_RNN[-1][-chunk_x+1:], [*dict_train[dict_keys][f'mfcc_{mfcc_num}'][i], dict_train[dict_keys]['mean_abs_val'][i], dict_train[dict_keys]['labels'][i]]]))       
                    
            for val in X_train_RNN:
                val[-1][-1] = -1
            
            model_RNN = modelRNN(np.array(X_train_RNN), np.array(y_train))

            dict_res[f'RNN_abs_{mfcc_num}_{chunk}_{chunk_x}'] = {}
            for dict_keys in list(dict_test.keys()):
                val_array = [[*[-1000]*mfcc_num, -1000, -1]]* (chunk_x)
                prev_pred = -1
                dict_res_tmp = dict_results_creator(dict_res[f'RNN_abs_{mfcc_num}_{chunk}_{chunk_x}'], dict_keys)
                dict_res_tmp['correct_breaths'] = dict_test[dict_keys]['n_breaths']

                for idx, val in enumerate(dict_test[dict_keys][f'mfcc_{mfcc_num}']):
                    dict_res_tmp['n_preds'] += 1
                    val_array = [*val_array[1:], [*val, dict_test[dict_keys]['mean_abs_val'][idx], -1]] 
                    pred = approx(model_RNN.predict(np.array([val_array])))
                    if pred == dict_test[dict_keys]['labels'][idx]:
                        dict_res_tmp['correct_preds'] += 1
                    if pred!=prev_pred:
                        dict_res_tmp['n_breaths'] += 1
                    prev_pred = pred
                    val_array[-1][-1] = pred

            model_BiRNN = modelBiRNN(np.array(X_train_RNN), np.array(y_train))

            dict_res[f'BiRNN_abs_{mfcc_num}_{chunk}_{chunk_x}'] = {}
            for dict_keys in list(dict_test.keys()):
                val_array = [[*[-1000]*mfcc_num, -1000, -1]]* (chunk_x)
                prev_pred = -1
                dict_res_tmp = dict_results_creator(dict_res[f'BiRNN_abs_{mfcc_num}_{chunk}_{chunk_x}'], dict_keys)
                dict_res_tmp['correct_breaths'] = dict_test[dict_keys]['n_breaths']

                for idx, val in enumerate(dict_test[dict_keys][f'mfcc_{mfcc_num}']):
                    dict_res_tmp['n_preds'] += 1
                    val_array = [*val_array[1:], [*val, dict_test[dict_keys]['mean_abs_val'][idx], -1]]  
                    pred = approx(model_BiRNN.predict(np.array([val_array])))
                    if pred == dict_test[dict_keys]['labels'][idx]:
                        dict_res_tmp['correct_preds'] += 1
                    if pred!=prev_pred:
                        dict_res_tmp['n_breaths'] += 1
                    prev_pred = pred
                    val_array[-1][-1] = pred
            
def main():
    train_path = "./NewClips/train"
    test_path = "./NewClips/test"
    results_dict = {}

    for chunk in CHUNK_ARRAY:
        train_dict_vals = {}
        test_dict_vals = {}
        create_vals(train_dict_vals, train_path, chunk)
        create_vals(test_dict_vals, test_path, chunk)
        create_train_test(train_dict_vals, test_dict_vals, results_dict, chunk)
        
        df = pd.DataFrame(results_dict)
        print(df)
        df.to_csv('./Final_result.csv')



if __name__ == '__main__':
    main()