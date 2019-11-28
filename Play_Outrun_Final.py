#!/usr/bin/env python3
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
import time

##################################
# LOADING DEEPLEARNING LIBRARIES #
##################################
import keras
import keras.backend as K
from keras import metrics
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Concatenate, Flatten, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

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
def f(q):
    i = 0
    while i < 10000:
        q.enqueue(int(random.randint(1,1000)))
        i = i + 1

def generate_model():
    input_shape = (220, 320,3)
    num_classes = 3
    
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    return model

def capture_return_decision(queue,model): 
    
    images = queue.return_queue()
    images = np.array(images)
    images =  np.swapaxes(images,0,2)
    

    return model.predict_proba(images.reshape(1,220, 320, 3))

def capture(graber,queue): 
    while True:
    #print("Comecando a gavar...")
        sct_img = graber.grab({'left': 715, 'top': 67, 'width': 640, 'height': 440})
        Img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        Img.thumbnail((320,220), Image.ANTIALIAS)
        Img = np.array(Img.convert('L'))/255
        Img = Img.T#Img.reshape(1,220,320,1)
        queue.enqueue(Img)
        time.sleep(0.15)
    
def replicate(source_device, target_device):
    for event in source_device.async_read_loop():
        #print('Man Generated Event',event)
        target_device.write_event(event)

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
    
    print('Loading Model...')
    model = generate_model()
    model.load_weights('/home/ormenesse/Documents/Reinforced_Outrun/model_3_outputs.h5')
    
    """
        SCREENSHOT DISPLAY
    """
    
    graber = mss.mss()
    
    #BaseManager.register('CircularQueue', CircularQueue)
    #manager = BaseManager()
    #queue = manager.CircularQueue(3)
    #p = Process(target=capture, args=[graber,queue])
    #p.daemon = True
    #p.start()
    queue = CircularQueue(3)
    queue_thread = threading.Thread(target=capture, args=(graber,queue))
    queue_thread.daemon = True
    queue_thread.start()
    

    """ REMEMBER THE THRESHOLDS 
    ['acelera', 'freia', 'direita', 'esquerda', 'cima', 'baixo']
       0.99      0.039      0.28       0.27       0       0.019
       1         1          1          -1         -1      1
    """

    print('Playing...')

    monitor_thread = threading.Thread(target=replicate, args=(dev, virtual_gamepad))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    print('Waiting all modules to initialize...')
    time.sleep(3)
    
    print('Start playing')
    while True:

        b = capture_return_decision(queue,model)
        # learning what to return
        freia = [1 if b[0][0] >= 0.10  else 0][0]

        if freia == 0:
            virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 0)
            virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 1)
        if freia == 1:
            virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 0)
            virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, freia)

        direcionalx = 0 
        if b[0][1] <= 0.1 and b[0][2] <= 0.11:
            direcionalx = 0
        elif b[0][1] > b[0][2]:
            direcionalx = 1
        elif b[0][1] < b[0][2]:
            direcionalx = -1
        else:
            direcionalx = 0

        direcionaly = 0
        #if b[0][4] >= 0.02:
        #    direcionaly = 1
        #if b[0][3] > 0.001:
        #    direcionaly = -1
        #working with directionals in the controller.
        virtual_gamepad.write(ecodes.EV_ABS,ecodes.ABS_HAT0X,direcionalx)
        virtual_gamepad.write(ecodes.EV_ABS,ecodes.ABS_HAT0Y,direcionaly)
        virtual_gamepad.syn()
        #print(direcionalx,direcionaly,freia)
        print(np.round(b,2))
        #time.sleep(0.05)
