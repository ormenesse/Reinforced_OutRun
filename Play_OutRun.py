#!/usr/bin/env python3
# Always add this line on the begginig 
# #chmod +0777 /dev/uinput

"""
xbox commands
A - type 1 (EV_KEY), code 304 (BTN_SOUTH), value 1
B - type 1 (EV_KEY), code 305 (BTN_EAST), value 1
X - type 1 (EV_KEY), code 307 (BTN_NORTH), value 1
Y - type 1 (EV_KEY), code 308 (BTN_WEST), value 1
D-UP -   type 3 (EV_ABS), code 17 (ABS_HAT0Y), value -1
D-DOWN - type 3 (EV_ABS), code 17 (ABS_HAT0Y), value 1
D-LEFT - type 3 (EV_ABS), code 16 (ABS_HAT0X), value -1
D-RIGHT - type 3 (EV_ABS), code 16 (ABS_HAT0X), value 1
L-A-UP - type 3 (EV_ABS), code 1 (ABS_Y), value -32262
L-A-D - type 3 (EV_ABS), code 1 (ABS_Y), value 32262
L-A-L - type 3 (EV_ABS), code 0 (ABS_X), value  -32262
L-A-R - type 3 (EV_ABS), code 0 (ABS_X), value 32262
R-A-UP -  type 3 (EV_ABS), code 4 (ABS_RY), value -32262
R-A-D -  type 3 (EV_ABS), code 4 (ABS_RY), value 32262
R-A-L -  type 3 (EV_ABS), code 3 (ABS_RX), value  -32262
R-A-R -  type 3 (EV_ABS), code 3 (ABS_RX), value 32262
L - type 1 (EV_KEY), code 310 (BTN_TL), value 1
R - type 1 (EV_KEY), code 311 (BTN_TR), value 1
LT - type 1 (EV_KEY), code 312 (BTN_TL2), value 1
RT - type 1 (EV_KEY), code 313 (BTN_TR2), value 1
SEL - type 1 (EV_KEY), code 314 (BTN_SELECT), value 1
START - type 1 (EV_KEY), code 315 (BTN_START), value 1
HOME - type 1 (EV_KEY), code 316 (BTN_MODE), value 0
BT_THUMBL - type 1 (EV_KEY), code 316 (BTN_MODE), value 1
BT_THUMBR - type 1 (EV_KEY), code 317 (BTN_THUMBL), value 1
BT_THUMBR - type 1 (EV_KEY), code 318 (BTN_THUMBR), value 1
"""

###########################
# LOADING EVDEV LIBRARIES #
###########################
import asyncio
import evdev
from evdev import list_devices, InputDevice, InputEvent, categorize, ecodes, UInput
import evdev

import pyautogui

##################################
# LOADING              LIBRARIES #
##################################


import threading
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import mss
import sched, time
import cv2
import pickle
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import subprocess
import os, sys
import re 

##################################
# LOADING DEEPLEARNING LIBRARIES #
##################################
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

##################################
# LOADING FUNCTIONS & CLASSES    #
##################################


class CircularQueue(object):
    #Constructor
    def __init__(self,maxSize=3):
        self.queue = list()
        self.head = 0
        self.maxSize = maxSize
    #Adding elements to the queue
    def enqueue(self,data):
        if len(self.queue) >= self.maxSize:
            self.queue.pop(0)
            self.queue.append(data)
        else:
            self.queue.append(data)
        return True
    def return_queue(self):
        return self.queue

def save_to_file(objeto, nome_arquivo):
    with open(nome_arquivo, 'wb') as output:
        pickle.dump(objeto, output, pickle.HIGHEST_PROTOCOL)


def load_file(nome_arquivo):
    with open(nome_arquivo, 'rb') as input:
        objeto = pickle.load(input)
    return objeto
    
##################################
#         OTHER FUNCTIONS        #
##################################

def demote(user_uid, user_gid):
    """Pass the function 'set_ids' to preexec_fn, rather than just calling
    setuid and setgid. This will change the ids for that subprocess only"""

    def set_ids():
        os.setgid(user_gid)
        os.setuid(user_uid)

    return set_ids

def generate_emulator_pid():

    #p = subprocess.Popen(['gens','/home/ormenesse/Documents/Reinforced_Outrun/OutRun (USA, Europe).md'],preexec_fn=demote(1000, 1000))
    p = subprocess.Popen(['gens','/home/ormenesse/Documents/Reinforced_Outrun/OutRun (USA, Europe).md'])
    # wait for the process to start
    time.sleep(0.15)

    # you gotta have xdotool installed in your machine

    output = subprocess.check_output(['xdotool','search','--pid',str(p.pid)])
    xw_id = str(output).split('\\n')[1]
    output = subprocess.run(['xwininfo','-id',xw_id],stdout=subprocess.PIPE)
    output = output.stdout.decode('utf-8')
    print(output)
    #get screens positions
    upperleftx = int(re.findall('Absolute upper-left X:  (\d+)',output)[0])#int(output.split('\n')[3][-4:])
    upperlefty = int(re.findall('Absolute upper-left Y:  (\d+)',output)[0])#int(output.split('\n')[4][-4:])

    # load quick save state
    
    pyautogui.press('f8')
    
    print('\n\nEnd Of Starting Emulator...')
    
    return p, upperleftx, upperlefty

def kill_process(p):

    p.kill()

def capture(graber,positionx,positiony):
    
    global queue
    
    while True:
    #print("Comecando a gavar...")
        sct_img = graber.grab({'left': positionx, 'top': positiony+40, 'width': 640, 'height': 440})
        Img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        Img.thumbnail((320,220), Image.ANTIALIAS)
        Img = np.array(Img.convert('L')).astype(np.uint8)
        #Img = Img.T
        queue.enqueue(Img)
        time.sleep(0.2)
    
def replicate(source_device, target_device):
    for event in source_device.async_read_loop():
        #print('Man Generated Event',event)
        target_device.write_event(event)
        
##################################
#           SCORE MODEL          #
##################################

#generate score model
def generate_scoring_model():
    
    model = tf.keras.models.load_model('./OCR_MODEL_OUTRUN/'+'modelo_captcha_outrun')
    
    return model

# transforming vector to numbers.
def vec_to_captcha(vec):
    text = []
    captcha_word = "0123456789"
    vec[vec < 0.5] = 0
    
    captcha_word = "0123456789"

    word_len = 8
    word_class = len(captcha_word)
    
    char_pos = vec.nonzero()[0]
    
    for i, ch in enumerate(char_pos):
        text.append(captcha_word[ch % word_class])
        
    return ''.join(text)

def return_only_numbers(i):
    return np.expand_dims(np.array(i[14:25,135:201]).astype(np.float), axis=-1)

def return_score(image,model):
    try:
        value = int(vec_to_captcha(model.predict(np.expand_dims(return_only_numbers(image),axis=0))[0]))
        if value >= 0:
            return value
        else:
            return 0
    except:
        return -1

# using OpenCV
import cv2, pytesseract
def return_score_cv_result(img):
    res = cv2.resize(img[14:25,135:201], dsize=(4*img[14:25,135:201].shape[1],4*img[14:25,135:201].shape[0]), interpolation=cv2.INTER_CUBIC)
    rest = pytesseract.image_to_string(cv2.blur(res,(5,5)),nice=1,config='--psm 7 --oem 3 digits')
    try:
        return int(rest)
    except:
        return 0

##################################
#         DECISION MODEL         #
##################################

# action model
def generate_model():
    input_shape = (220, 320,3)
    num_classes = 14
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x/255,input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    #model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.add(tf.keras.layers.Dense(num_classes))
    
    return model

def get_controler_action(predict):
    
    # plausible positions I got when gaming
    # 100000*comandos['acelera']+10000*comandos['freia']+1000*comandos['direita']+100*comandos['esquerda']+10*comandos['cima']+comandos['baixo']
    comands = [     0,    100,   1000,  10100,  11000, 100000, 100001, 100010, 100100, 101000, 101001, 101010, 110100, 111000]
    
    argument = predict.argmax()
    
    cmd_number = comands[argument]
    
    if cmd_number == 0:
        
        return [0, 0, 0, 0, 0, 0]
    
    else:
        
        baixo = cmd_number // 10**1 % 10
        
        cima = cmd_number // 10**2 % 10
        
        esquerda = cmd_number // 10**3 % 10
        
        direita = cmd_number // 10**4 % 10
        
        freia = cmd_number // 10**5 % 10
        
        acelera = cmd_number // 10**6 % 10
        
        return [acelera, freia, direita, esquerda, cima, baixo]
    
def capture_return_decision(images,model): 
    
    # predict command
    imgs = np.array(images).astype(np.uint8)
    imgs = np.expand_dims(imgs, axis=0)
    imgs = np.swapaxes(imgs,1,2)
    imgs = np.swapaxes(imgs,2,3)
    
    return model.predict(imgs), imgs

        
##################################
#              MAIN              #
##################################        

# global Queue
queue = CircularQueue(3)
#global command variable
commands = [0, 0, 0, 0, 0, 0]

if __name__ == '__main__':
    
    """
    HOOK GAMEPAD.
    """
    # Listen to events from the following device.
    print('Will Inject on device:',list_devices()[0])
    dev = InputDevice( list_devices()[0] )

    # Then create our own devices for our own events.
    # (It is possible to write events directly to the original device, but that has caused me mysterious problems in the past.)
    virtual_gamepad = UInput.from_device(dev, name="virtual-gamepad")

    # Now we monopolize the original devices.
    dev.grab()

    # Sends all events from the source to the target, to make our virtual keyboard behave as the original.
    


    print('Virtual Gamepad created. \nReady to play videogames!')
    
    """
        LOADING MODEL
    """
    
    print('\n\nLoading Models...')
    score_model = generate_scoring_model()
    model = generate_model()
    # pre trained model
    model.load_weights('/home/ormenesse/Documents/Reinforced_Outrun/Action_Encoded_Model/modelo_outrun_encoded_learning_best_mse.hdf5')
    
    """
        SCREENSHOT DISPLAY
    """

    
    graber = mss.mss()
    
    # starting emulator
    p, upperleftx, upperlefty = generate_emulator_pid()
    
    # starting image queue
    #queue = CircularQueue(3)
    queue_thread = threading.Thread(target=capture, args=(graber,upperleftx, upperlefty))
    queue_thread.daemon = True
    queue_thread.start()
    
    #controller.
    monitor_thread = threading.Thread(target=replicate, args=(dev, virtual_gamepad))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    print('Waiting all modules to initialize...')
    time.sleep(1)
    
    pyautogui.press('f8')
    print('\n\nStart playing...')
    
    errors = 0
    
    imgs = []
    
    while True:
        
        try:
            
            images          = queue.return_queue()
            #score_f         = return_score(images[2],score_model)
            #score_i         = return_score(images[0],score_model)
            score_f         = return_score_cv_result(images[2])
            score_i         = return_score_cv_result(images[0])
            score           = score_f-score_i
            b, treated_imgs = capture_return_decision(images,model)
            commands        = get_controler_action(b)
            
            imgs.append((images.copy(),score_f,score_i,commands))
            
            # learning what to return
            # commands = [acelera, freia, direita, esquerda, cima, baixo]

            if commands[0] == 1 and commands[1] == 0:  # accelerate 1 and brake 0
                virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 1)
                virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 0)
            elif commands[1] == 1 and commands[0] == 0: # accelerate 0 and brake 0
                virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 0)
                virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 1)
            else: # else do nothing
                virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 1)
                virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 0)

            # left righ decision
            direcionalx = 0 
            if commands[2] == 1 and commands[3] == 0:
                direcionalx = 1 # go right
            elif commands[2] == 0 and commands[3] == 1:
                direcionalx = -1 # go left
            else: # go straight ahead
                direcionalx = 0

            
            direcionaly = 0
            if commands[4] == 1 and commands[5] == 0:
                direcionaly = -1 # up
                virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_EAST, 1)
            elif commands[4] == 0 and commands[5] == 1:
                direcionaly = 1 # down
                virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_EAST, 1)
            else: # nothing
                direcionalx = 0
                virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_EAST, 0) # always 0

            #working with directionals in the controller.
            virtual_gamepad.write(ecodes.EV_ABS,ecodes.ABS_HAT0X,direcionalx)
            virtual_gamepad.write(ecodes.EV_ABS,ecodes.ABS_HAT0Y,direcionaly)
            virtual_gamepad.syn()
            
            print('commands ' + str(commands),'argmax',str(b.argmax()),' game score', score,'score_f',  score_f, 'score_i',score_i)
            
        except Exception as e: 
            
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
            errors = errors + 1
        
        save_to_file(imgs, './Analyse_Data/imgs.pkl')
        
        if errors == 5:
            kill_process(p)
            break
            
        time.sleep(0.1)
