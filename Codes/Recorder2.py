import PySimpleGUI as sg
import pyaudio
import numpy as np
import librosa
import librosa.display 
import tensorflow as tf
import math
import time 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse

# VARS CONSTS:
_VARS = {'trainWindow' : False,
         'window': False, #must be
         'stream': False, #must be
         'model':False, #used model
         'array':[], #array with the history
         'train_array':[],
         'train_flag':0,
         'flag':0, #inhale or exhale
         'exhales':0, #number of exhales
         'time':0, #start time
         'time_single':0, #time of one exhale or inhale
         'text':'Please start breathing', #String for presenting text
         'text_top': "",
         'fig_agg':False,
         'plt_fig':False,
         'if_plot':False} 

# INITS:
CHUNK = 4096  # Samples: 1024,  512, 256, 128
RATE = 44100  # Equivalent to Human Hearing at 40 kHz
N_MFCC = 40 #MFFC number
N_LOOPS = 3 #Number of training loops
BREATHING_TIME = 1.5 #Time of exhale and inhale in seconds
EPOCHS = 30
CALLBACK = tf.keras.callbacks.EarlyStopping(monitor = 'loss', min_delta = 1e-5, patience = 5, restore_best_weights = True)

pAud = pyaudio.PyAudio()

AppFont = 'Any 16'
sg.theme('Default1')

# FUNCTIONS:

#Creating Windows
def trainWindow():    
    train_layout = [[sg.Text(_VARS['text'], key='-TrainText-')],
                    [sg.Button('Start', font=AppFont)]]
    _VARS['trainWindow'] = sg.Window('trainWindow', train_layout, element_justification="center", finalize=True)

def mainWindow():
    layout = [[sg.Text(_VARS['text_top'], key='-Timer-')],
    [sg.Canvas(key='-Plot-')],
    [sg.Text(_VARS['text'], key = '-RR-')],
    [sg.Button('Listen', font=AppFont),
           sg.Button('Exit', font=AppFont)]]
    _VARS['window'] = sg.Window('RR', layout, element_justification="center", finalize=True)
#-----------------------------------------------------------------------------------------------------

# RR functions
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
        _VARS['time_single'] = time.time()

    elif len(array)>=3:
        if (array[-3] == array[-2] == array[-1]) and array[-1]!= _VARS['flag']:
            _VARS['flag'] = array[-1]
            RRprinter(array[-1], end)
            _VARS['time_single'] = time.time()
#-----------------------------------------------------------------------------------------------------

#Train
def callbackTrain(in_data, frame_count, time_info, status):
    data = (np.frombuffer(in_data, dtype=np.float32))
    mfccs_features = librosa.feature.mfcc(y=data, sr=RATE/2, n_mfcc=N_MFCC)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0).reshape(N_MFCC)
    _VARS['train_array'].append([mfccs_scaled_features, _VARS['train_flag']])
    return (in_data, pyaudio.paContinue)

def startTrainWindow():
    _VARS['stream'] = pAud.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=int(RATE/2),
                                input=True,
                                frames_per_buffer=CHUNK,
                                stream_callback=callbackTrain)
    _VARS['stream'].start_stream()

def trainLoopFunction(i, train_flag, text1, text2):
    _VARS['train_flag'] = train_flag
    _VARS['text'] = f'Click Start and start {text1} for {BREATHING_TIME} seconds. Done:{i+1}/{N_LOOPS}'
    trainWindow()
    while True:
        event, values = _VARS['trainWindow'].read(timeout=200)
        if event == sg.WIN_CLOSED:
            stop()
            exit()
        if event == 'Start':
            _VARS['time'] = time.time()
            startTrainWindow()
        if _VARS['time'] != 0:
            time_left = BREATHING_TIME - (time.time() - _VARS['time'])
            _VARS['text'] = f'{text2} for {time_left:.4f} seconds'
            _VARS['trainWindow']['-TrainText-'].update(_VARS['text'])
            if time.time() - _VARS['time'] > BREATHING_TIME:
                stop()
                _VARS['time'] = 0
                break
    _VARS['trainWindow'].close()

def trainNewModel():
    X = []
    y = []
    for val, label in _VARS['train_array']:
        X.append(val)
        y.append(label)
    X_train = np.array(X)
    y_train = np.array(y)
    _VARS['model'].fit(X_train, y_train, batch_size=1, epochs=EPOCHS, callbacks=[CALLBACK])
#-----------------------------------------------------------------------------------------------------

#Main application functions
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def stop():
    if _VARS['stream']:
        _VARS['stream'].stop_stream()
        _VARS['stream'].close()

def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    end = time.time()
    mfccs_features = librosa.feature.mfcc(y=data, sr=RATE/2, n_mfcc=N_MFCC)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    test_data = mfccs_scaled_features.reshape(1,mfccs_scaled_features.shape[0])
    pred = _VARS['model'].predict(test_data)
    _VARS['array'].append(approx(pred))
    RRcounter(_VARS['array'], end)
    process_time = int(end - _VARS['time'])
    if _VARS['if_plot'] == True:
        if _VARS['fig_agg'] != False:
            _VARS['fig_agg'].get_tk_widget().forget()
            plt.clf()
        plt.plot(data)
        plt.ylim([-0.002,0.002])
        single_hale = time.time() - _VARS['time_single']
        if _VARS['flag'] == 1:
            _VARS['text_top'] = (f'Whole time : {process_time}, You are EXHALING for: {single_hale:.2f} seconds')
        else:
            _VARS['text_top'] = (f'Whole time : {process_time}, You are INHALING for: {single_hale:.2f} seconds')
        _VARS['fig_agg'] = draw_figure(_VARS['window']['-Plot-'].TKCanvas, _VARS['plt_fig'])
    _VARS['window']['-RR-'].update(_VARS['text'])
    _VARS['window']['-Timer-'].update(_VARS['text_top'])
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
#-----------------------------------------------------------------------------------------------------
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model_1", help="name of the model")
    parser.add_argument('--plot', type=int, default=1, choices=[0,1], help="plot the graph? 1-yes, 0-no")
    parser.add_argument('--train', type=int, default=1, choices=[0,1], help="do additional training? 1-yes, 0-no")
    
    args = parser.parse_args()
    # MAIN LOOP
    _VARS['model'] = tf.keras.models.load_model(f".\Models\{args.model}.h5")
    _VARS['model'].compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
    if args.plot == 1:
        _VARS['if_plot'] = args.plot
        _VARS['plt_fig'] = plt.figure(figsize=[6,4])
    
    # Training LOOP 
    if args.train == 1:
        for i in range(N_LOOPS):
            trainLoopFunction(i, 0, "inhaling", "Inhale")
            trainLoopFunction(i, 1, "exhaling", "Exhale")
        trainNewModel()

    #Main LOOP
    _VARS['text'] = f'Click Listen and start breathing'
    mainWindow()
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