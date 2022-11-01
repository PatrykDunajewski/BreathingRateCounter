import PySimpleGUI as sg
import pyaudio
import numpy as np
import librosa
import librosa.display 
import pandas as pd
import tensorflow as tf
from sklearn import metrics
import math
import time 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse

# VARS CONSTS:
_VARS = {'window': False, #must be
         'stream': False, #must be
         'model':False, #used model
         'array':[], #array with the history
         'flag':0, #inhale or exhale
         'exhales':0, #number of exhales
         'time':0, #start time
         'text':'Please start breathing', #String for presenting text
         'fig_agg':False,
         'plt_fig':False,
         'if_plot':False} 

# pysimpleGUI INIT:
AppFont = 'Any 16'
sg.theme('Default1')
layout = [[sg.Canvas(key='-Plot-')],
    [sg.Text(_VARS['text'], key = '-RR-')],
    [sg.Button('Listen', font=AppFont),
           sg.Button('Exit', font=AppFont)]]
_VARS['window'] = sg.Window('RR', layout, finalize=True)


# PyAudio INIT:
CHUNK = 4096  # Samples: 1024,  512, 256, 128
RATE = 44100  # Equivalent to Human Hearing at 40 kHz

pAud = pyaudio.PyAudio()

# FUNCTIONS:

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def approx(val):
    if val >=0.5:
        return math.ceil(val)
    else:
        return math.floor(val)

def RRprinter(val, end):
    if(val == 1):
        _VARS['exhales']+=1
        print('Exhale')
        process_time = int(end - _VARS['time'])
        if process_time == 0:
            process_time = 1
        respiratory_rate = int(_VARS['exhales']/process_time * 60)
        print(f"Number of exhales: {_VARS['exhales']}, time: {process_time}, RR: {respiratory_rate}")
        _VARS['text'] = f"Number of exhales: {_VARS['exhales']}, time: {process_time}, RR: {respiratory_rate}"
    else:
        print('Inhale')

def RRcounter(array, end):
    if len(array)==1:
        RRprinter(array[0], end)
        _VARS['flag'] = array[0]

    elif len(array)>=3:
        if (array[-3] == array[-2] == array[-1]) and array[-1]!= _VARS['flag']:
            _VARS['flag'] = array[-1]
            RRprinter(array[-1], end)


def stop():
    if _VARS['stream']:
        _VARS['stream'].stop_stream()
        _VARS['stream'].close()


def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    end = time.time()
    mfccs_features = librosa.feature.mfcc(y=data, sr=RATE/2, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    test_data = mfccs_scaled_features.reshape(1,mfccs_scaled_features.shape[0])
    pred = _VARS['model'].predict(test_data)
    _VARS['array'].append(approx(pred))
    RRcounter(_VARS['array'], end)
    if _VARS['if_plot'] == True:
        if _VARS['fig_agg'] != False:
            _VARS['fig_agg'].get_tk_widget().forget()
            plt.clf()
        plt.plot(data)
        plt.ylim([-0.002,0.002])
        process_time = int(end - _VARS['time'])
        if _VARS['flag'] == 1:
            plt.title(f'Time : {process_time}, Exhale')
        else:
            plt.title(f'Time : {process_time}, Inhale')
        _VARS['fig_agg'] = draw_figure(_VARS['window']['-Plot-'].TKCanvas, _VARS['plt_fig'])
    _VARS['window']['-RR-'].update(_VARS['text'])
    return (in_data, pyaudio.paContinue)


def listen():
    _VARS['window'].FindElement('Listen').Update(disabled=True)
    _VARS['stream'] = pAud.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=int(RATE/2),
                                input=True,
                                frames_per_buffer=CHUNK,
                                stream_callback=callback)
    _VARS['stream'].start_stream()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model_1", help="name of the model")
    parser.add_argument('--plot', type=bool, default=True, help="plot the graph")
    
    args = parser.parse_args()
    # MAIN LOOP
    _VARS['model'] = tf.keras.models.load_model(f".\Models\{args.model}.h5")
    _VARS['model'].compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
    if args.plot == True:
        _VARS['if_plot'] = args.plot
        _VARS['plt_fig'] = plt.figure(figsize=[6,4])
    while True:
        event, values = _VARS['window'].read(timeout=200)
        if event == sg.WIN_CLOSED or event == 'Exit':
            stop()    
            pAud.terminate()        
            break
        if event == 'Listen':
            _VARS['time'] = time.time()
            listen()

    _VARS['window'].close()


if __name__ == "__main__":
    main()