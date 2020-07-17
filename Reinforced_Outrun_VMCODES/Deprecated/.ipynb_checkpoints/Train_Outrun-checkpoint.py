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
import gc

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
    time.sleep(3)

    # you gotta have xdotool installed in your machine

    output = subprocess.check_output(['xdotool','search','--pid',str(p.pid)])
    print('Emulator Screen Id:',str(output))
    xw_id = re.findall('(\d+)',str(output))[1]#str(output).split('\\n')[1]
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
    
    #defining global variables
    global queue
    global commands
    global model 
    global capture_screen
    
    count_capture = 0
    
    while True:
        
        if capture_screen:
            
            sct_img = graber.grab({'left': positionx, 'top': positiony+40, 'width': 640, 'height': 440})
            Img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
            Img.thumbnail((320,220), Image.ANTIALIAS)
            Img = np.array(Img.convert('L')).astype(np.uint8)



            if count_capture == 5:

                commands, b  = get_controller_action_random()
                
                count_capture = 0
                
                time.sleep(0.04) # assumption

            else:
                try:
                    
                    b = capture_return_decision([j[0] for j in queue[-5:]])
                    print(b)
                    commands = get_controller_action_model(b)
                    
                except:
                    
                    time.sleep(0.05) # assumption
                    
                    commands = [0, 0, 0, 0, 0]
                    
                    b = np.zeros((1,14))

                count_capture = count_capture + 1

            # Add image do Queue
            queue.append((Img.copy(),b[0].copy(),time.time()))

        #time.sleep(0.01) # it takes almost 0.2s
    
def replicate(source_device, target_device):
    for event in source_device.async_read_loop():
        #print('Man Generated Event',event)
        target_device.write_event(event)
        
def apply_model_controller_output():
    global capture_screen
    global commands
    while True:
        if capture_screen:
            write_to_controller_output(commands)
        
def write_to_controller_output(command):
    global virtual_gamepad
    # [accelerate    brake        right      left       gearup&down]
    # [10000*i[0] +  1000*i[1]    +100*i[2]  +10*i[3]     +1*i[4]    for i in cmds]
    
    if command[0] == 1 and command[1] == 0:  # accelerate 1 and brake 0
        virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 1)
        virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 0)
    elif command[1] == 1 and command[0] == 0: # accelerate 0 and brake 0
        virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 0)
        virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 1)
    else: # else do nothing
        virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 0)
        virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 0)

    # left righ decision
    direcionalx = 0 
    if command[2] == 1 and command[3] == 0:
        direcionalx = 1 # go right
    elif command[2] == 0 and command[3] == 1:
        direcionalx = -1 # go left
    else: # go straight ahead
        direcionalx = 0


    # Gear
    if command[4] == 1:
        virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_EAST, 1)
    else:
        virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_EAST, 0)

    #working with directionals in the controller.
    virtual_gamepad.write(ecodes.EV_ABS,ecodes.ABS_HAT0X,direcionalx)
    #virtual_gamepad.write(ecodes.EV_ABS,ecodes.ABS_HAT0Y,direcionaly)
    virtual_gamepad.syn()
        
##################################
#           SCORE MODEL          #
##################################

#generate score model
def generate_scoring_model():
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x/255,input_shape=(9, 8, 1)))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    
    model.load_weights('./OCR_POINTS_MODEL_OUTRUN/'+'model_score_outrun.h5')
    
    return model

def return_score(image,model):
    limit_numbers_1 = np.arange(200,136,-8)
    limit_numbers_2 = np.arange(192,128,-8)
    
    score = 0
    
    for j in range(len(limit_numbers_1)):
        
        i = model.predict(np.expand_dims(np.expand_dims(image[15:24,limit_numbers_2[j]:limit_numbers_1[j]],axis=-1),axis=0))[0]
        
        score = score + i.argmax()*(10**j)
    
    return score

##################################
#           TIME MODEL           #
##################################

#generate score model
def generate_time_model():
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x/255,input_shape=(16, 17, 1)))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=20, activation='sigmoid'))
    
    model.load_weights('./OCR_TIME_GAME_OVER_MODEL/'+'model_time_outrun.h5')
    
    return model

# useful methods
def captcha_to_vec(captcha):
    
    # parameters

    captcha_word = "0123456789"

    word_len = 2
    word_class = len(captcha_word)

    char_indices = dict((c, i) for i,c in enumerate(captcha_word))
    indices_char = dict((i, c) for i,c in enumerate(captcha_word))
    vector = np.zeros(word_len * word_class)
    
    for i,ch in enumerate(captcha):
        idex = i * word_class + char_indices[ch]
        vector[idex] = 1
    return vector

def vec_to_captcha(vec):
    
    # parameters

    captcha_word = "0123456789"

    word_len = 2
    word_class = len(captcha_word)

    char_indices = dict((c, i) for i,c in enumerate(captcha_word))
    indices_char = dict((i, c) for i,c in enumerate(captcha_word))
    vector = np.zeros(word_len * word_class)
    
    text = []
    vec[vec < 0.5] = 0
        
    char_pos = vec.nonzero()[0]
    
    for i, ch in enumerate(char_pos):
        text.append(captcha_word[ch % word_class])
    return ''.join(text)


def return_time(image,model):
    
    return int(vec_to_captcha(model.predict(np.expand_dims(np.expand_dims(image[7:23,56:73],axis=-1),axis=0))[0]))

##################################
#         DECISION MODEL         #
##################################

# action model
def generate_model():
    input_shape = (220, 320,5)
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

def retrain_model():
    
    global queue
    global model
    global score_model
    global capture_screen
    global model_not_compiled
    
    capture_screen = False
    
    print('Training will have: ',len(queue),'samples over 5 epochs.')
    
    #Applying Q-Learning
    # learning rate
    LR = 0.1
    
    cmds = []
    frame_imgs = []
    score = []
    time = []
    print('Starting to organize data before training... Nº', 0, 'of', len(queue),'samples.', end='\r')
    for i in np.arange(5,len(queue)-5,1):
        print('Starting to organize data before training... Nº', i, 'of', len(queue),'samples.', end='\r')
        # (Img.copy(),b[0].copy(),time.time())
        frame_imgs.append(model_capture_return_images([j[0] for j in queue[i-5:i]]))
        cmds.append(queue[i][1])
        # appending score
        score_i = return_score(queue[i][0],score_model)
        score_f = return_score(queue[i+5][0],score_model)
        score.append(score_f-score_i)
        
        time.append(queue[i+5][2]-queue[i][2])
        # Q - Learning Happening Here
        if score[-1] <= 0:
            score[-1] = -1
        else:
            score[-1] = score[-1]/1000

        cmds[-1][cmds[-1].argmax()] = cmds[-1][cmds[-1].argmax()] + LR*score[-1]
    
    print('\n')
    print('Mean time between samples:',np.mean(time),'sec.')
    print('Mean Score between samples:',np.mean(score),'points.')
    print('Maximum Train Score:',return_score(queue[-1][0],score_model))
    
    frame_imgs = np.array(frame_imgs)
    
    cmds = np.array(cmds)
    
    #Train model
    INIT_LR = 1e-3
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer='adam',
                  metrics=['mse'])
    model_not_compiled = False
    
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,verbose=0, mode='min')

    model.fit(frame_imgs[:,:,:,:], cmds,
              batch_size=32,
              epochs=5,
              shuffle = True,
              validation_data=(frame_imgs[:,:,:,:], cmds),
              callbacks=[earlyStopping])
    test_score_loss = model.evaluate(frame_imgs[:,:,:,:], cmds, verbose=0)
    
    #model.save("./Action_Encoded_Model/modelo_outrun.hdf5")
    model.save_weights("./Action_Encoded_Model/modelo_outrun.hdf5")
    ################################
    # reloading model              #
    # compiled models are too slow #
    ################################
    model = generate_model()
    # pre trained model
    model.load_weights('./Action_Encoded_Model/modelo_outrun.hdf5')
    
    queue = []
    queue_img = []
    del cmds, frame_imgs, score, time
    
    # restarting
    pyautogui.press('f8')
    
    print('Test loss:', test_score_loss[0])
    
    capture_screen = True
    
    # Cleaning Memory
    gc.collect()
    
    return True

def get_controller_action_model(predict):
    
    # plausible positions I got when gaming
    # [accelerate    brake        right      left       gearup&down]
    # [10000*i[0] +  1000*i[1]    +100*i[2]  +10*i[3]     +1*i[4]    for i in cmds]
    cmds = [0, 10, 11, 100, 101, 1000, 1010, 1100, 10000, 10001, 10010, 10011, 10100, 10101]
    
    argument = predict.argmax()
    
    cmd_number = cmds[argument]
    
    if cmd_number == 0:
        
        return [0, 0, 0, 0, 0]
    
    else:
        
        acelera = cmd_number // 10**4 % 10
        
        freia = cmd_number // 10**3 % 10
        
        direita = cmd_number // 10**2 % 10
        
        esquerda = cmd_number // 10**1 % 10
        
        gear = cmd_number // 10**0 % 10
        
        return [acelera, freia, direita, esquerda, gear]
    
def get_controller_action_random():
    
    # plausible positions I got when gaming
    # [accelerate    brake        right      left       gearup&down]
    # [10000*i[0] +  1000*i[1]    +100*i[2]  +10*i[3]     +1*i[4]    for i in cmds]
    cmds = [0, 10, 11, 100, 101, 1000, 1010, 1100, 10000, 10001, 10010, 10011, 10100, 10101]
    
    cmd_number = np.random.choice(cmds,1)[0]
    
    b = np.zeros((1,14))
    b[:,cmds.index(cmd_number)] = 1
    
    if cmd_number == 0:
        
        return [0, 0, 0, 0, 0], b
    
    else:
        
        acelera = cmd_number // 10**4 % 10
        
        freia = cmd_number // 10**3 % 10
        
        direita = cmd_number // 10**2 % 10
        
        esquerda = cmd_number // 10**1 % 10
        
        gear = cmd_number // 10**0 % 10
        
        return [acelera, freia, direita, esquerda, gear], b
    
def capture_return_decision(images):
    
    global model
    
    # predict command
    imgs = np.array(images).astype(np.uint8)
    imgs = np.expand_dims(imgs, axis=0)
    imgs = np.swapaxes(imgs,1,2)
    imgs = np.swapaxes(imgs,2,3)
    #print('capture_return_decision shape',imgs.shape)
    return model.predict(imgs)

def model_capture_return_images(images): 
    
    # predict command
    imgs = np.array(images).astype(np.uint8)
    #imgs = np.expand_dims(imgs, axis=0)
    imgs = np.swapaxes(imgs,0,1)
    imgs = np.swapaxes(imgs,1,2)
    #print('model_capture_return_images shape',imgs.shape)
    return imgs
        
##################################
#              MAIN              #
##################################        

# global Queue
queue = []
# global command variable
commands = [0, 0, 0, 0, 0]
# global capture
capture_screen = False
model_not_compiled = True

if __name__ == '__main__':
    
    """
    HOOK GAMEPAD.
    """
    # Listen to events from the following device.
    print('Will Inject on device:',list_devices()[0])
    dev = InputDevice( list_devices()[0] )

    # Then create our own devices for our own events.
    # (It is possible to write events directly to the original device, but that has caused me mysterious problems in the past.)
    virtual_gamepad = UInput.from_device(dev, name="Microsoft X-Box 360 pad")#name="virtual-gamepad")

    # Now we monopolize the original devices.
    dev.grab()

    # Sends all events from the source to the target, to make our virtual keyboard behave as the original.
    


    print('Virtual Gamepad created. \nReady to play videogames!')
    
    """
        LOADING MODEL
    """
    
    print('\n\nLoading Models...')
    score_model = generate_scoring_model()
    time_model = generate_time_model()
    model = generate_model()
    # pre trained model
    model.load_weights('./Action_Encoded_Model/modelo_outrun.hdf5')
    
    """
        SCREENSHOT DISPLAY
    """

    
    graber = mss.mss()
    
    # starting emulator
    p, upperleftx, upperlefty = generate_emulator_pid()
    
    # starting image queue
    #queue = CircularQueue(3)
    
    # capture Thread
    queue_thread = threading.Thread(target=capture, args=(graber,upperleftx, upperlefty))
    #queue_thread.daemon = True
    queue_thread.start()
    
    # controller Thread
    monitor_thread = threading.Thread(target=replicate, args=(dev, virtual_gamepad))
    #monitor_thread.daemon = True
    monitor_thread.start()
    
    # Apply Commands Thread
    apply_cmds_thread = threading.Thread(target=apply_model_controller_output)
    #monitor_thread.daemon = True
    apply_cmds_thread.start()
    
    
    print('Waiting all modules to initialize...')
    
    pyautogui.press('f8')
    capture_screen = True
    time.sleep(1)
    print('\n\nStart playing...')
    
    #####################
    # control variables #
    #####################
    end_epoch = 0
    random_file_number = str(np.round(np.random.random()*1000,0))
    print('\n\n\nRandom File Number Generated:', random_file_number,'\n\n\n')
    _time_ = 80
    
    while True:
        
        _time_ = return_time(queue[-1][0],time_model)
        
        if _time_ <= 1: # Finding game over not yet implemented.
            print('Saving File...')
            end_epoch = end_epoch + 1
            save_to_file(queue, './ANALYSE_DATA/fila_treino_'+random_file_number+'_'+str(end_epoch)+'.pkl')
            retrain_model()
            gc.collect()
            
        time.sleep(0.3)