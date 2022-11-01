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
import pandas as pd

import pygame

pygame.init()
pygame.mixer.init()

# VARS CONSTS:

# INITS:
CHUNK = 4096  # Samples: 1024,  512, 256, 128
RATE = 44100  # Equivalent to Human Hearing at 40 kHz
N_MFCC = 20 #MFFC number
N_LOOPS = 3 #Number of training loops
BREATHING_TIME = 1.5 #Time of exhale and inhale in seconds
EPOCHS = 30
CALLBACK = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 1e-5, patience = 10, restore_best_weights = True)
PLOTSIZE = 0.002
REFTIME = 50
EWMA_WINDOW = 5
PLOT_STEPS = 20
CHANNELS = 1
BACK_CHUNK = 20
BATCH = 4
MODEL_PATH = ".\Models\Model_RNN_Final.h5"

_GUI_VARS = {'start_time':0,
             'stop_time_begin':0,
             'stop_time':0,
             'text':"Please click one of the buttons belowe to start the application.",
             'text_top':"Welcome to the Respiration Rate application.",
             'if_app_begin':True,
             'guiWindow':False,
             'if_basic_model':True
}

_SETTINGS_VARS = {'settingWindow':False,
                 'n_loops':N_LOOPS,
                 'breathing_time':BREATHING_TIME,
                 'epochs':EPOCHS,
                 'time_text':"You will stop time by yourself.",
                 'time_flag':0
}

_TRAIN_VARS = {'trainWindow' : False,
                'train_array':[],
                'train_flag':0,
                'train_time':0,
                'train_text':"",
                'train_break_flag':0
}

_VARS = {'stream':False, #must be
         'model':False, #used model
         'array':[], #array with the history
         'flag':0, #inhale or exhale
         'exhales':0, #number of exhales
         'time_single':0, #time of one exhale or inhale
         'fig_agg':False,
         'pltFig':False,
         'xData':False,
         'yData':False,
         'audioData':np.array([]),
         'proper_pred_array':[-1] * PLOT_STEPS,
         'master_volume':0.5,
         'mic_sens':0,
         'emwa_array':[],
         'max_plot_size':PLOTSIZE,
         'main_plot_array':[0] * PLOT_STEPS,
         'number_of_iters':-PLOT_STEPS,
         'vals_array': [[*[-1000]*N_MFCC, -1000, -1]] * (BACK_CHUNK - 1)
} 

pAud = pyaudio.PyAudio()
AppFont = 'Any 16'
sg.theme('Default1')

# FUNCTIONS:

#Creating Windows
def guiWindow():
    gui_layout = [[sg.Text(_GUI_VARS['text_top'], key='-Timer-')],
                [sg.Canvas(key='-Plot-')],
                [sg.Text("Current iteration: "),
                 sg.Slider(range=(0,_VARS['number_of_iters']+PLOT_STEPS), default_value=_VARS['number_of_iters']+PLOT_STEPS, orientation='h', key='-PlotSlider-')],
                [sg.Text(_GUI_VARS['text'], key = '-RR-')],
                 [sg.Button('Listen', font=AppFont), sg.Button('Train', font=AppFont), sg.Button('Stop', font=AppFont, disabled=True),
                  sg.Button('Clear', font=AppFont, disabled=_GUI_VARS['if_app_begin']), sg.Button('Clear model', font=AppFont, disabled=_GUI_VARS['if_basic_model']),
                  sg.Button('Exit', font=AppFont)],
                  [sg.Text("Volume: "), sg.Slider(range=(0,100), default_value=_VARS['master_volume']*100, orientation="h", key='-VolSlider-')],
                  [sg.Text("Microphone sensitivity: "), sg.Slider(range=(-100,100), default_value=_VARS['mic_sens'], orientation="h", key='-SensSlider-')]]
    _GUI_VARS['guiWindow'] = sg.Window('RR App', gui_layout, relative_location=(-200,-200), element_justification="center", finalize=True)

def settingsTrainWindow():
    settings_layout = [[sg.Text("Number of breathing loops: "), sg.Slider(range=(1, 10), default_value=N_LOOPS, orientation="h", key="-NBreathing-")],
                    [sg.Radio("Controlling yourself", 1, enable_events=True, default=True, key="-R1-"), sg.Radio("Constant time", 1, enable_events=True, default=False, key="-R2-")],
                    [sg.Text(_SETTINGS_VARS['time_text'], key="-TimeSettingText-"), 
                    sg.Slider(range=(5, 50), default_value=BREATHING_TIME*10, orientation="h", visible=False, key="-TBreathing-")],
                    [sg.Text("Number of epochs: "), sg.Slider(range=(1, 100), default_value=EPOCHS, orientation="h", key="-NEpochs-")],
                    [sg.Button("Confirm", font=AppFont), sg.Button("Cancel", font=AppFont)]]
    _SETTINGS_VARS['settingWindow'] = sg.Window('Train settings', settings_layout, element_justification="center", finalize=True)

def trainWindow(i):    
    train_layout = [[sg.Text(_TRAIN_VARS['train_text'], key='-TrainText-')],
                    [sg.Canvas(key='-TrainPlot-')],
                    [sg.Button('Start', font=AppFont), sg.Button('Cancel', font=AppFont)]]
    train_yourself_layout = [[sg.Text(_TRAIN_VARS['train_text'], key='-TrainText-')],
                    [sg.Canvas(key='-TrainPlot-')],
                    [sg.Button('Start', font=AppFont), sg.Button('Stop', font=AppFont, disabled=True), sg.Button('Cancel', font=AppFont)]]
    layouts_array = [train_layout, train_yourself_layout]
    _TRAIN_VARS['trainWindow'] = sg.Window('Train window', layouts_array[i], relative_location=(-200,-200), element_justification="center", finalize=True)

def trainModelWindow():
    train_model_layout = [[sg.Text("Please wait a moment. The model is in the process of training. When the training is completed, this window will automatically close.")]]
    _TRAIN_VARS['trainWindow'] = sg.Window('Model train window', train_model_layout, element_justification="center", finalize=True)
#-----------------------------------------------------------------------------------------------------

#GUI Functions
def createGUIWindow():
    stop_flag = 1
    guiWindow()
    _VARS['xData'] = np.linspace(_VARS['number_of_iters'], _VARS['number_of_iters']+PLOT_STEPS, num=PLOT_STEPS, dtype=int)
    _VARS['yData'] = np.zeros(PLOT_STEPS)
    drawPlot(_GUI_VARS['guiWindow']['-Plot-'], -_VARS['max_plot_size']/10)
    if _GUI_VARS['if_app_begin'] == False:
        updatePlot(np.array(_VARS['main_plot_array'][-PLOT_STEPS:]), -_VARS['max_plot_size']/10, 0, 0)
    while True:
        event, values = _GUI_VARS['guiWindow'].read(timeout=REFTIME)
        if event == sg.WIN_CLOSED or event == 'Exit':
            stop()
            break
        
        if event == 'Listen':
            _GUI_VARS['if_app_begin'] = False
            stop_flag = 0
            if _GUI_VARS['start_time'] == 0:  
                _GUI_VARS['start_time'] = time.time()
            else:
                _GUI_VARS['stop_time'] += time.time() - _GUI_VARS['stop_time_begin']
                _GUI_VARS['stop_time_begin'] = 0
            listen()
            
        if _VARS['audioData'].size != 0 and stop_flag == 0:
            _VARS['number_of_iters']+=1
            y = ewmaMy(np.mean(np.abs(_VARS['audioData'])))
            _VARS['main_plot_array'] = [*_VARS['main_plot_array'], y]
            _VARS['xData'] = np.linspace(_VARS['number_of_iters'], _VARS['number_of_iters']+PLOT_STEPS, num=PLOT_STEPS, dtype=int)
            _GUI_VARS['guiWindow']['-PlotSlider-'].update(range=(0, _VARS['number_of_iters']+PLOT_STEPS), value = _VARS['number_of_iters']+PLOT_STEPS)
            _GUI_VARS['guiWindow']['-RR-'].update(_GUI_VARS['text'])
            _GUI_VARS['guiWindow']['-Timer-'].update(_GUI_VARS['text_top'])
            updatePlot(np.array(_VARS['main_plot_array'][-PLOT_STEPS:]), -_VARS['max_plot_size']/10, 0, 0)
        
        if _GUI_VARS['if_app_begin'] == False and stop_flag == 1:
            _VARS['xData'] = np.linspace(values['-PlotSlider-']-PLOT_STEPS, values['-PlotSlider-'], num=PLOT_STEPS, dtype=int)
            updatePlot(np.array(_VARS['main_plot_array'][int(values['-PlotSlider-']):int(values['-PlotSlider-']+PLOT_STEPS)]), -_VARS['max_plot_size']/10, 0, 
            int(_VARS['number_of_iters']+PLOT_STEPS-values['-PlotSlider-']))
        
        if event == 'Train':
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
            _GUI_VARS['guiWindow']['-PlotSlider-'].update(disabled=False)
            if _GUI_VARS['if_basic_model'] == False:
                _GUI_VARS['guiWindow']['Clear model'].update(disabled=False)
            stop()
            
        if event == 'Clear':
            clear_vals()

        if event == 'Clear model':
            _VARS['model'] = tf.keras.models.load_model(MODEL_PATH)
            _VARS['model'].compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
            _GUI_VARS['guiWindow']['Clear model'].update(disabled=True)
            _GUI_VARS['if_basic_model'] = True

        _VARS['master_volume'] = values['-VolSlider-']/100
        _VARS['mic_sens'] = values['-SensSlider-']
#-----------------------------------------------------------------------------------------------------
#Plot Functions 
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='bottom', fill='both', expand=1)
    return figure_canvas_agg

def drawPlot(window, y_lim_min):
    _VARS['audioData'] = np.array([])
    plt.style.use('ggplot')
    _VARS['pltFig'] = plt.figure()
    plt.plot(_VARS['xData'], _VARS['yData'], '-k', linewidth=2)
    plt.ylim(y_lim_min, _VARS['max_plot_size'])
    _VARS['fig_agg'] = draw_figure(window.TKCanvas, _VARS['pltFig'])

def updatePlot(data, y_lim_min, if_train, slider):
    plt.cla()
    if if_train == 0:
        pred = [*_VARS['proper_pred_array'][-PLOT_STEPS-slider:-1-slider], _VARS['proper_pred_array'][-1]]
        for idx, val in enumerate(_VARS['proper_pred_array'][-PLOT_STEPS-slider:-1-slider]):
            x, y = zip((_VARS['xData'][idx], data[idx]),(_VARS['xData'][idx+1], data[idx+1]))
            if pred[idx+1] == -1:
                seg_color = 'k'
            elif pred[idx+1] == 1:
                seg_color = 'r' 
            else:
                seg_color = 'g'
            plt.plot(x, y, '-', color=seg_color, linewidth=2)
    else:
        plt.plot(_VARS['xData'], data, '-k', linewidth=1)
    plt.ylim(y_lim_min, _VARS['max_plot_size'])
    _VARS['fig_agg'].draw()

def clearPlot(window, y_lim_min):
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.cla()
    _VARS['audioData'] = np.array([])
    plt.plot(_VARS['xData'], _VARS['yData'], '-k', linewidth=2)
    plt.ylim(y_lim_min, _VARS['max_plot_size'])
    _VARS['fig_agg'] = draw_figure(window.TKCanvas, _VARS['pltFig'])

#-----------------------------------------------------------------------------------------------------
def ewmaMy(y):
    if len(_VARS['emwa_array']) <= EWMA_WINDOW - 1:
        _VARS['emwa_array'].append(y)
    else:
         _VARS['emwa_array'] = [*_VARS['emwa_array'][1:], y]

    numbers_series = pd.Series(_VARS['emwa_array'])
    moving_average = round(numbers_series.ewm(alpha=0.5, adjust=False).mean(), len(_VARS['emwa_array']))
    return moving_average.tolist()[-1]

def getMaxPlotSize(y):
    if y > _VARS['max_plot_size']:
        _VARS['max_plot_size'] = y

def micSensConvertion(data):    
    # mic_sens_corr = np.power(10.0,(_VARS['mic_sens']+100)/20.0)
    # data = ((data/np.power(2.0,15))*5.25)*(mic_sens_corr) 
    mic_sens_corr = 1 + _VARS['mic_sens']/100
    data = data * mic_sens_corr
    return data

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
        _VARS['exhales']+=1
        process_time = end - (_GUI_VARS['start_time'] + _GUI_VARS['stop_time'])
        if int(process_time) == 0:
            process_time = 1
        respiratory_rate = int(_VARS['exhales']/process_time * 60)
        _GUI_VARS['text'] = f"Number of exhales: {_VARS['exhales']}, time: {process_time:.2f}, RR: {respiratory_rate}"

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
            _VARS['proper_pred_array'][-1] = _VARS['flag']
        else:
            if _VARS['time_single'] == 0:
                _VARS['time_single'] = time.time()
    
    _VARS['proper_pred_array'].append(_VARS['flag'])
#-----------------------------------------------------------------------------------------------------

#Train
def createSettingsTrainWindow():
    do_train = 0
    _SETTINGS_VARS['n_loops'] = N_LOOPS
    _SETTINGS_VARS['breathing_time'] = BREATHING_TIME
    _SETTINGS_VARS['epochs'] = EPOCHS
    _SETTINGS_VARS['time_flag'] = 1
    _SETTINGS_VARS['time_text'] = "You will stop time by yourself."
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
            _SETTINGS_VARS['time_flag'] = 1
            _SETTINGS_VARS['time_text'] = "You will stop time by yourself."
            _SETTINGS_VARS['settingWindow']['-TimeSettingText-'].update(_SETTINGS_VARS['time_text'])
            _SETTINGS_VARS['settingWindow']['-TBreathing-'].update(visible=False)
        if values["-R2-"] == True:
            _SETTINGS_VARS['time_flag'] = 0
            _SETTINGS_VARS['time_text'] = "Time of one inhale or exhale (*0.1 secound): "
            _SETTINGS_VARS['settingWindow']['-TimeSettingText-'].update(_SETTINGS_VARS['time_text'])
            _SETTINGS_VARS['settingWindow']['-TBreathing-'].update(visible=True)
    _SETTINGS_VARS['settingWindow'].close()

    if do_train == 0:
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
            _GUI_VARS['if_basic_model'] = False
            _TRAIN_VARS['trainWindow'].close()
            sg.Popup(f'Max accuracy from training: {np.max(history.history["val_accuracy"]):.2f}; mean accuracy: {np.mean(history.history["val_accuracy"]):.2f}')

    createGUIWindow()


def callbackTrain(in_data, frame_count, time_info, status):
    _VARS['audioData'] = micSensConvertion(np.frombuffer(in_data, dtype=np.float32))
    getMaxPlotSize(np.mean(np.abs(_VARS['audioData'])))
    mfccs_features = librosa.feature.mfcc(y=_VARS['audioData'], sr=RATE, n_mfcc=N_MFCC)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0).reshape(N_MFCC)
    val = np.mean(np.abs(_VARS['audioData']))
    _TRAIN_VARS['train_array'].append([[*mfccs_scaled_features, val, _TRAIN_VARS['train_flag']], _TRAIN_VARS['train_flag']])
    # _TRAIN_VARS['train_array'].append([[_TRAIN_VARS['train_flag'], np.mean(np.abs(_VARS['audioData'])), rms_val, *mfccs_scaled_features[[5, 17]]], _TRAIN_VARS['train_flag']])
    return (in_data, pyaudio.paContinue)

def startTrainWindow():
    _TRAIN_VARS['trainWindow']['Start'].update(disabled=True)
    _VARS['stream'] = pAud.open(format=pyaudio.paFloat32,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK,
                                stream_callback=callbackTrain)
    _VARS['stream'].start_stream()

def trainLoopFunction(i, train_flag, text1, text2):
    _TRAIN_VARS['train_flag'] = train_flag
    time_flag = _SETTINGS_VARS['time_flag']
    if time_flag == 0:
        _TRAIN_VARS['train_text'] = f'Click Start and start {text1} for {_SETTINGS_VARS["breathing_time"]:.2f} seconds. Done:{i+1}/{_SETTINGS_VARS["n_loops"]}'
    else:
        _TRAIN_VARS['train_text'] = f'Click Start and start {text1}. Time is determined by you. Done:{i+1}/{_SETTINGS_VARS["n_loops"]}'
    trainWindow(time_flag)

    _VARS['xData'] = np.linspace(0,CHUNK, num=CHUNK, dtype=int)
    _VARS['yData'] = np.zeros(CHUNK)
    drawPlot(_TRAIN_VARS['trainWindow']['-TrainPlot-'], -_VARS['max_plot_size'])
    while True:
        event, values = _TRAIN_VARS['trainWindow'].read(timeout=REFTIME)
        if event == sg.WIN_CLOSED or event == 'Cancel':
            stop()
            _TRAIN_VARS['train_break_flag'] = 1
            break
        if event == 'Start':
            _TRAIN_VARS['train_time'] = time.time()
            startTrainWindow()
            if time_flag == 1:    
                _TRAIN_VARS['trainWindow']['Start'].update(disabled=True)
                _TRAIN_VARS['trainWindow']['Stop'].update(disabled=False)
        if event == 'Stop':
            stop()
            _TRAIN_VARS['train_time'] = 0
            break

        if _VARS['audioData'].size != 0:
            updatePlot(_VARS['audioData'], -_VARS['max_plot_size'], 1, 0)
            if time_flag == 1:
                time_passed = time.time() - _TRAIN_VARS['train_time']
                _TRAIN_VARS['train_text'] = f'You are {text1} for {time_passed:.4f} seconds. Please click Stop button when you are done.'
                _TRAIN_VARS['trainWindow']['-TrainText-'].update(_TRAIN_VARS['train_text'])

        if _TRAIN_VARS['train_time'] != 0 and time_flag == 0:
            time_left = _SETTINGS_VARS["breathing_time"] - (time.time() - _TRAIN_VARS['train_time'])
            _TRAIN_VARS['train_text'] = f'{text2} for {time_left:.4f} seconds'
            _TRAIN_VARS['trainWindow']['-TrainText-'].update(_TRAIN_VARS['train_text'])
            if time_left <= 0:
                stop()
                _TRAIN_VARS['train_time'] = 0
                break
    _TRAIN_VARS['trainWindow'].close()

def trainNewModel():
    X = [[*[-1000]*N_MFCC, -1000, -1]] * (BACK_CHUNK - 1) 
    X_train = []
    y =[]
    y_train = []

    for val, label in _TRAIN_VARS['train_array']:
        X.append(val)
        y.append(label)

    for i in range(np.array(X).shape[0]-BACK_CHUNK):
        X_train.append(np.array(X)[i:i+BACK_CHUNK])
        X_train[-1][-1][-1] = -1
        y_train.append(np.array(y)[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    history = _VARS['model'].fit(X_train, y_train, batch_size=BATCH, validation_split=0.2, epochs=_SETTINGS_VARS['epochs'], callbacks=[CALLBACK])
    _VARS['vals_array'] = [[*[-1000]*N_MFCC, -1000, -1]] * (BACK_CHUNK - 1)
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
    _VARS['master_volume'] = 0.5
    _VARS['array'] = []
    _VARS['vals_array'] = [[*[-1000]*N_MFCC, -1000, -1]] * (BACK_CHUNK - 1)
    _VARS['emwa_array'] = []
    _VARS['max_plot_size'] = PLOTSIZE
    _VARS['number_of_iters'] = -PLOT_STEPS 
    _GUI_VARS['guiWindow']['-PlotSlider-'].update(range=(0, _VARS['number_of_iters']+PLOT_STEPS), value = _VARS['number_of_iters']+PLOT_STEPS)
    _VARS['xData'] = np.linspace(_VARS['number_of_iters'], PLOT_STEPS+_VARS['number_of_iters'], num=PLOT_STEPS, dtype=int)
    _VARS['yData'] = np.zeros(PLOT_STEPS)
    _VARS['proper_pred_array'] = [-1] * PLOT_STEPS
    _VARS['main_plot_array'] = [0] * PLOT_STEPS
    _GUI_VARS['if_app_begin'] = True


    clearPlot(_GUI_VARS['guiWindow']['-Plot-'], 0)

    _GUI_VARS['guiWindow']['Stop'].update(disabled=True)
    _GUI_VARS['guiWindow']['Clear'].update(disabled=True)
    _GUI_VARS['guiWindow']['Train'].update(disabled=False)
    _GUI_VARS['guiWindow']['Listen'].update(disabled=False)

def callback(in_data, frame_count, time_info, status):
    _VARS['audioData'] = micSensConvertion(np.frombuffer(in_data, dtype=np.float32))
    getMaxPlotSize(np.mean(np.abs(_VARS['audioData'])))
    end = time.time()

    mfccs_features = librosa.feature.mfcc(y=_VARS['audioData'], sr=RATE, n_mfcc=N_MFCC)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0).reshape(N_MFCC)
    val = np.mean(np.abs(_VARS['audioData']))
    _VARS['vals_array'].append([*mfccs_scaled_features, val, -1])
    # _VARS['vals_array'].append([-1, np.mean(np.abs(_VARS['audioData'])), rms_val, *mfccs_scaled_features[[5,17]]])
    data = np.array(_VARS['vals_array'])[-BACK_CHUNK:]
    data = data.reshape(1, data.shape[0], data.shape[1])
    pred = _VARS['model'].predict(data)
    _VARS['array'].append(approx(pred))
    _VARS['vals_array'][-1][-1] = approx(pred)
    RRcounter(_VARS['array'], end)
    process_time = end - (_GUI_VARS['start_time'] + _GUI_VARS['stop_time'])
    single_hale = time.time() - _VARS['time_single']
    if _VARS['flag'] == 1:
        _GUI_VARS['text_top'] = (f'Whole time : {process_time:.2f}, You are EXHALING for: {single_hale:.2f} seconds')
    else:
        _GUI_VARS['text_top'] = (f'Whole time : {process_time:.2f}, You are INHALING for: {single_hale:.2f} seconds')

    return (in_data, pyaudio.paContinue)

def listen():
    _GUI_VARS['guiWindow']['-PlotSlider-'].update(disabled=True)
    _GUI_VARS['guiWindow']['Stop'].update(disabled=False)
    _GUI_VARS['guiWindow']['Clear model'].update(disabled=True)
    _GUI_VARS['guiWindow']['Clear'].update(disabled=True)
    _GUI_VARS['guiWindow']['Train'].update(disabled=True)
    _GUI_VARS['guiWindow']['Listen'].update(disabled=True)
    _VARS['stream'] = pAud.open(format=pyaudio.paFloat32,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK,
                                stream_callback=callback)
    _VARS['stream'].start_stream()
#-----------------------------------------------------------------------------------------------------
    

def main():
    # MAIN LOOP
    _VARS['model'] = tf.keras.models.load_model(MODEL_PATH)
    _VARS['model'].compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="adam")
    createGUIWindow()
if __name__ == "__main__":
    main()