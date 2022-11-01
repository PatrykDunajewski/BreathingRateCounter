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
from playsound import playsound

import pygame

pygame.init()
pygame.mixer.init()

# VARS CONSTS:

# INITS:
CHUNK = 2048  # Samples: 1024,  512, 256, 128
RATE = 44100  # Equivalent to Human Hearing at 40 kHz
N_MFCC = 40 #MFFC number
N_LOOPS = 3 #Number of training loops
BREATHING_TIME = 1.5 #Time of exhale and inhale in seconds
EPOCHS = 30
CALLBACK = tf.keras.callbacks.EarlyStopping(monitor = 'loss', min_delta = 1e-5, patience = 10, restore_best_weights = True)
PLOTSIZE = 0.002
REFTIME = 50
MIN_TRAIN_TIME = 0.3


_GUI_VARS = {'start_time':0,
             'stop_time_begin':0,
             'stop_time':0,
             'text':"Please click one of the buttons above to launch the application.",
             'text_top': "Welcome to the Respiration Rate application.",
             'not_trained':True,
             'guiWindow':False
}

_SETTINGS_VARS = {'settingWindow':False,
                 'n_loops':N_LOOPS,
                 'breathing_time':BREATHING_TIME,
                 'epochs':EPOCHS,
                 'time_text': "Time of one inhale or exhale (*0.1 secound): ",
                 'time_flag':0
}

_TRAIN_VARS = {'trainWindow' : False,
                'train_array':[],
                'train_flag':0,
                'train_time':0,
                'train_text':"",
                'train_break_flag':0,
                'time_dec': 0
}

_VARS = {'stream': False, #must be
         'model':False, #used model
         'array':[], #array with the history
         'flag':0, #inhale or exhale
         'exhales':0, #number of exhales
         'time_single':0, #time of one exhale or inhale
         'fig_agg':False,
         'pltFig': False,
         'xData': False,
         'yData': False,
         'audioData': np.array([]),
         'master_volume': 0.5
} 


pAud = pyaudio.PyAudio()
AppFont = 'Any 16'
sg.theme('Default1')

# FUNCTIONS:

#Creating Windows
def guiWindow():
    gui_layout = [[sg.Text(_GUI_VARS['text_top'], key='-Timer-')],
                [sg.Canvas(key='-Plot-')],
                [sg.Text(_GUI_VARS['text'], key = '-RR-')],
                 [sg.Button('Listen', font=AppFont), sg.Button('Train', font=AppFont), sg.Button('Stop', font=AppFont, disabled=True),
                  sg.Button('Clear', font=AppFont, disabled=_GUI_VARS['not_trained']), sg.Button('Exit', font=AppFont)],
                  [sg.Text("Volume: "), sg.Slider(range=(0,100), default_value=_VARS['master_volume']*100, orientation="horizontal")]]
    _GUI_VARS['guiWindow'] = sg.Window('RR App', gui_layout, relative_location=(-200,-200), element_justification="center", finalize=True)

def settingsTrainWindow():
    settings_layout = [[sg.Text("Number of breathing loops: "), sg.Slider(range=(2, 10), default_value=N_LOOPS, orientation="horizontal", key="-NBreathing-")],
                    [sg.Radio("Constant time", 1, enable_events=True, default=True, key="-R1-"), sg.Radio("Decreasing time", 1, enable_events=True, default=False, key="-R2-")],
                    [sg.Text(_SETTINGS_VARS['time_text'], key="-TimeSettingText-"), sg.Slider(range=(5, 50), default_value=BREATHING_TIME*10, orientation="horizontal", key="-TBreathing-")],
                    [sg.Text("Number of epochs: "), sg.Slider(range=(1, 100), default_value=EPOCHS, orientation="horizontal", key="-NEpochs-")],
                    [sg.Button("Confirm", font=AppFont), sg.Button("Cancel", font=AppFont)]]
    _SETTINGS_VARS['settingWindow'] = sg.Window('Train settings', settings_layout, element_justification="center", finalize=True)

def trainWindow():    
    train_layout = [[sg.Text(_TRAIN_VARS['train_text'], key='-TrainText-')],
                    [sg.Canvas(key='-TrainPlot-')],
                    [sg.Button('Start', font=AppFont), sg.Button('Cancel', font=AppFont)]]
    _TRAIN_VARS['trainWindow'] = sg.Window('Train window', train_layout, relative_location=(-200,-200), element_justification="center", finalize=True)


def trainModelWindow():
    train_model_layout = [[sg.Text("Please wait a moment. The model is in the process of training. When the training is completed, this window will automatically close.")]]
    _TRAIN_VARS['trainWindow'] = sg.Window('Model train window', train_model_layout, element_justification="center", finalize=True)
#-----------------------------------------------------------------------------------------------------

#GUI Functions
def createGUIWindow():
    stop_flag = 1
    guiWindow()
    drawPlot(_GUI_VARS['guiWindow']['-Plot-'])
    while True:
        event, values = _GUI_VARS['guiWindow'].read(timeout=REFTIME)
        if event == sg.WIN_CLOSED or event == 'Exit':
            stop()
            break
        
        if event == 'Listen':
            stop_flag = 0
            if _GUI_VARS['start_time'] == 0:  
                _GUI_VARS['start_time'] = time.time()
            else:
                _GUI_VARS['stop_time'] += time.time() - _GUI_VARS['stop_time_begin']
                _GUI_VARS['stop_time_begin'] = 0
            listen()
            
        if _VARS['audioData'].size != 0 and stop_flag == 0:
            updatePlot(_VARS['audioData'], _GUI_VARS['guiWindow']['-Plot-'])
            _GUI_VARS['guiWindow']['-RR-'].update(_GUI_VARS['text'])
            _GUI_VARS['guiWindow']['-Timer-'].update(_GUI_VARS['text_top'])
        
        if event == 'Train':
            _GUI_VARS['not_trained']=False
            if _GUI_VARS['stop_time_begin'] == 0:
                _GUI_VARS['stop_time_begin'] == time.time()
            _GUI_VARS['guiWindow'].close() 
            createSettingsTrainWindow()
        
        if event == 'Stop':
            _GUI_VARS['stop_time_begin'] = time.time()
            stop_flag = 1
            _GUI_VARS['guiWindow']['Stop'].update(disabled=True)
            _GUI_VARS['guiWindow']['Clear'].update(disabled=False)
            _GUI_VARS['guiWindow']['Train'].update(disabled=False)
            _GUI_VARS['guiWindow']['Listen'].update(disabled=False)
            stop()
            
        if event == 'Clear':
            clear_vals()

        
        _VARS['master_volume'] = values[0]/100
#-----------------------------------------------------------------------------------------------------
#Plot Functions 
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def drawPlot(window):
    _VARS['audioData'] = np.array([])
    plt.style.use('ggplot')
    _VARS['xData'] = np.linspace(0, CHUNK, num=CHUNK, dtype=int)
    _VARS['yData'] = np.zeros(CHUNK)
    _VARS['pltFig'] = plt.figure()
    plt.plot(_VARS['xData'], _VARS['yData'], '--k')
    plt.ylim(-PLOTSIZE, PLOTSIZE)
    _VARS['fig_agg'] = draw_figure(window.TKCanvas, _VARS['pltFig'])

def updatePlot(data, window):
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.cla()
    plt.clf()
    plt.plot(_VARS['xData'], data, '--k')
    plt.ylim(-PLOTSIZE, PLOTSIZE)
    _VARS['fig_agg'] = draw_figure(window.TKCanvas, _VARS['pltFig'])

def clearPlot(window):
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.cla()
    plt.clf()
    _VARS['audioData'] = np.array([])
    plt.plot(_VARS['xData'], _VARS['yData'], '--k')
    plt.ylim(-PLOTSIZE, PLOTSIZE)
    _VARS['fig_agg'] = draw_figure(window.TKCanvas, _VARS['pltFig'])


#-----------------------------------------------------------------------------------------------------

# RR functions
def approx(val):
    if val >=0.5:
        return math.ceil(val)
    else:
        return math.floor(val)

def RRprinter(val, end):
    if(val == 1):
        sound = pygame.mixer.Sound(".\\AppSounds\\boop_part.wav")
        sound.set_volume(_VARS['master_volume'])
        sound.play()
        # playsound(".\\AppSounds\\boop_part.wav")
        _VARS['exhales']+=1
        process_time = int(end - (_GUI_VARS['start_time'] + _GUI_VARS['stop_time']))
        if process_time == 0:
            process_time = 1
        respiratory_rate = int(_VARS['exhales']/process_time * 60)
        _GUI_VARS['text'] = f"Number of exhales: {_VARS['exhales']}, time: {process_time}, RR: {respiratory_rate}"

def RRcounter(array, end):
    if len(array)==1:
        RRprinter(array[0], end)
        _VARS['flag'] = array[0]
        _VARS['time_single'] = time.time()

    elif len(array)>=2:
        if (array[-2] == array[-1]) and array[-1]!= _VARS['flag']:
            _VARS['flag'] = array[-1]
            RRprinter(array[-1], end)
            _VARS['time_single'] = time.time()
        else:
            if _VARS['time_single'] == 0:
                _VARS['time_single'] = time.time()
#-----------------------------------------------------------------------------------------------------

#Train
def createSettingsTrainWindow():
    conf_val = sg.Popup("Are you sure you want to train model?", button_type=1)
    if conf_val == "Yes":
        do_train = 0
        conf_default = sg.Popup("Would you like to use default options?", button_type=1)
        if conf_default == "No":
            _TRAIN_VARS['time_dec'] = 0
            _SETTINGS_VARS['time_flag'] = 0
            _SETTINGS_VARS['time_text'] = "Time of one inhale or exhale (*0.1 secound): "
            settingsTrainWindow()
            while True:
                event, values = _SETTINGS_VARS['settingWindow'].read()
                if event == sg.WIN_CLOSED or event == "Cancel":
                    do_train = 1
                    break
                if event == "Confirm":
                    _SETTINGS_VARS['n_loops'] = int(values["-NBreathing-"])
                    _SETTINGS_VARS['breathing_time'] = float(values["-TBreathing-"]/10)
                    _SETTINGS_VARS['epochs'] = int(values["-NEpochs-"])
                    break
                if values["-R1-"] == True:
                    _SETTINGS_VARS['time_flag'] = 0
                    _SETTINGS_VARS['time_text'] = "Time of one inhale or exhale (*0.1 secound): "
                    _SETTINGS_VARS['settingWindow']['-TimeSettingText-'].update(_SETTINGS_VARS['time_text'])


                if values["-R2-"] == True:
                    _SETTINGS_VARS['time_flag'] = 1
                    _SETTINGS_VARS['time_text'] = "Max time of one inhale or exhale (*0.1 secound): "
                    _SETTINGS_VARS['settingWindow']['-TimeSettingText-'].update(_SETTINGS_VARS['time_text'])
            _SETTINGS_VARS['settingWindow'].close()

        else:
            _SETTINGS_VARS['n_loops'] = N_LOOPS
            _SETTINGS_VARS['breathing_time'] = BREATHING_TIME
            _SETTINGS_VARS['epochs'] = EPOCHS
            _SETTINGS_VARS['time_flag'] = 0
            _TRAIN_VARS['time_dec'] = 0
        
        if do_train == 0:
            if _SETTINGS_VARS['time_flag'] == 1:
                _TRAIN_VARS['time_dec'] = (_SETTINGS_VARS['breathing_time'] - MIN_TRAIN_TIME) / (_SETTINGS_VARS['n_loops'] - 1)
            _TRAIN_VARS['train_break_flag'] = 0
            _TRAIN_VARS['train_array'] = []
            for i in range(_SETTINGS_VARS['n_loops']):
                trainLoopFunction(i, 0, "inhaling", "Inhale")
                if _TRAIN_VARS['train_break_flag'] == 1: 
                    break
                trainLoopFunction(i, 1, "exhaling", "Exhale")
                if _TRAIN_VARS['train_break_flag'] == 1: 
                    break
            if _TRAIN_VARS['train_break_flag'] == 0: 
                trainModelWindow()
                history = trainNewModel()
                _TRAIN_VARS['trainWindow'].close()
                sg.Popup(f'Max accuracy from training: {np.max(history.history["accuracy"]):.2f}; mean accuracy: {np.mean(history.history["accuracy"]):.2f}')
    
    createGUIWindow()



def callbackTrain(in_data, frame_count, time_info, status):
    data = (np.frombuffer(in_data, dtype=np.float32))
    _VARS['audioData'] = data
    mfccs_features = librosa.feature.mfcc(y=data, sr=RATE/2, n_mfcc=N_MFCC)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0).reshape(N_MFCC)
    _TRAIN_VARS['train_array'].append([mfccs_scaled_features, _TRAIN_VARS['train_flag']])
    return (in_data, pyaudio.paContinue)

def startTrainWindow():
    _TRAIN_VARS['trainWindow']['Start'].update(disabled=True)
    _VARS['stream'] = pAud.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=int(RATE/2),
                                input=True,
                                frames_per_buffer=CHUNK,
                                stream_callback=callbackTrain)
    _VARS['stream'].start_stream()

def trainLoopFunction(i, train_flag, text1, text2):
    time_in_loop = _SETTINGS_VARS["breathing_time"] - i * _TRAIN_VARS['time_dec']
    _TRAIN_VARS['train_flag'] = train_flag
    _TRAIN_VARS['train_text'] = f'Click Start and start {text1} for {time_in_loop:.2f} seconds. Done:{i+1}/{_SETTINGS_VARS["n_loops"]}'
    trainWindow()
    drawPlot(_TRAIN_VARS['trainWindow']['-TrainPlot-'])
    while True:
        event, values = _TRAIN_VARS['trainWindow'].read(timeout=REFTIME)
        if event == sg.WIN_CLOSED or event == 'Cancel':
            stop()
            _TRAIN_VARS['train_break_flag'] = 1
            break
        if event == 'Start':
            _TRAIN_VARS['train_time'] = time.time()
            startTrainWindow()

        if _VARS['audioData'].size != 0:
            updatePlot(_VARS['audioData'], _TRAIN_VARS['trainWindow']['-TrainPlot-'])

        if _TRAIN_VARS['train_time'] != 0:
            time_left = time_in_loop - (time.time() - _TRAIN_VARS['train_time'])
            _TRAIN_VARS['train_text'] = f'{text2} for {time_left:.4f} seconds'
            _TRAIN_VARS['trainWindow']['-TrainText-'].update(_TRAIN_VARS['train_text'])
            if time_left <= 0:
                stop()
                _TRAIN_VARS['train_time'] = 0
                break
    _TRAIN_VARS['trainWindow'].close()

def trainNewModel():
    X = []
    y = []
    for val, label in _TRAIN_VARS['train_array']:
        X.append(val)
        y.append(label)
    X_train = np.array(X)
    y_train = np.array(y)
    history = _VARS['model'].fit(X_train, y_train, batch_size=1, epochs=_SETTINGS_VARS['epochs'], callbacks=[CALLBACK])
    return history

#-----------------------------------------------------------------------------------------------------

#Main application functions

def stop():
    if _VARS['stream']:
        _VARS['stream'].stop_stream()
        _VARS['stream'].close()

def clear_vals():
    _GUI_VARS['start_time'] = 0
    _GUI_VARS['stop_time'] = 0
    _VARS['exhales'] = 0
    _VARS['time_single'] = 0
    _GUI_VARS['text'] = "Time and number of exhales have been cleared"
    _GUI_VARS['guiWindow']['-RR-'].update(_GUI_VARS['text'])
    _GUI_VARS['text_top'] = "Welcome to the Respiration Rate application."
    _GUI_VARS['guiWindow']['-Timer-'].update(_GUI_VARS['text_top'])
    clearPlot(_GUI_VARS['guiWindow']['-Plot-'])

    _GUI_VARS['guiWindow']['Stop'].update(disabled=True)
    _GUI_VARS['guiWindow']['Clear'].update(disabled=True)
    _GUI_VARS['guiWindow']['Train'].update(disabled=False)
    _GUI_VARS['guiWindow']['Listen'].update(disabled=False)

def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    _VARS['audioData'] = data
    end = time.time()
    mfccs_features = librosa.feature.mfcc(y=data, sr=RATE/2, n_mfcc=N_MFCC)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    test_data = mfccs_scaled_features.reshape(1,mfccs_scaled_features.shape[0])
    pred = _VARS['model'].predict(test_data)
    _VARS['array'].append(approx(pred))
    RRcounter(_VARS['array'], end)
    process_time = int(end - (_GUI_VARS['start_time'] + _GUI_VARS['stop_time']))
    single_hale = time.time() - _VARS['time_single']
    if _VARS['flag'] == 1:
        _GUI_VARS['text_top'] = (f'Whole time : {process_time}, You are EXHALING for: {single_hale:.2f} seconds')
    else:
        _GUI_VARS['text_top'] = (f'Whole time : {process_time}, You are INHALING for: {single_hale:.2f} seconds')
    return (in_data, pyaudio.paContinue)

def listen():
    _GUI_VARS['guiWindow']['Stop'].update(disabled=False)
    _GUI_VARS['guiWindow']['Clear'].update(disabled=True)
    _GUI_VARS['guiWindow']['Train'].update(disabled=True)
    _GUI_VARS['guiWindow']['Listen'].update(disabled=True)
    _VARS['stream'] = pAud.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=int(RATE/2),
                                input=True,
                                frames_per_buffer=CHUNK,
                                stream_callback=callback)
    _VARS['stream'].start_stream()
#-----------------------------------------------------------------------------------------------------
    

def main():
    # MAIN LOOP
    _VARS['model'] = tf.keras.models.load_model(f".\Models\model_2048.h5")
    _VARS['model'].compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
    createGUIWindow()
if __name__ == "__main__":
    main()