#!/usr/bin/env python3
# Always add this line on the begginig 
# #chmod +0777 /dev/uinput

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
from XboxController import *

##################################
# LOADING DEEPLEARNING LIBRARIES #
##################################
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

##################################
# LOADING FUNCTIONS & CLASSES    #
##################################



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
    time.sleep(1)

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
    
    #defining global variables
    global queue
    global commands
    global controller
    
    while True:
    #print("Comecando a gavar...")
    
        time_1 = time.time()
        
        sct_img = graber.grab({'left': positionx, 'top': positiony+40, 'width': 640, 'height': 440})
        Img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        Img.thumbnail((320,220), Image.ANTIALIAS)
        Img = np.array(Img.convert('L')).astype(np.uint8)
        
        comandos = controller.read()
        
        direita = [1 if comandos[0] == 1 else 0][0]
        esquerda = [1 if comandos[0] == -1 else 0][0]
        
        #           accelerate    brake        right      left       gear up&down
        commands = [ comandos[2], comandos[3], direita , esquerda , comandos[4] ]
        
        queue.append((Img.copy(),commands.copy(),time.time()))
        
        time_2 = time.time()
        
        if time_2 - time_1 < 0.2:
            
            time.sleep(0.2-(time_2 - time_1))
            
def capture_controller():
    
    #defining global variables
    global controller
    global actions_analysis
    
    while True:
    #print("Comecando a gavar...")
    
        action = controller.read()
        
        if str(actions_analysis[-1][0]) != str(action):
            
            actions_analysis.append((action,time.time()))
        
            #print(len(actions_analysis))
    
def replicate(source_device, target_device):
    for event in source_device.async_read_loop():
        #print('Man Generated Event',event)
        target_device.write_event(event)

##################################
#              MAIN              #
##################################        

from gameover_model import *
import tensorflow as tf

# global Queue
queue = []

#global command variable
commands = []

# global read XboxController
controller = XboxController()

# actions_analysis
actions_analysis = [(controller.read(),time.time())]

random_file_number = str(int(np.round(np.random.random()*1000,0)))

if __name__ == '__main__':
    
    # Generating Time Model
    time_model = generate_time_model()
    
    print('Virtual Gamepad created. \nReady to play videogames!')
    
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
    
    
    #queue = CircularQueue(3)
    controller_thread = threading.Thread(target=capture_controller, args=())
    controller_thread.daemon = True
    #controller_thread.start()
    
    print('Waiting all modules to initialize...')
    time.sleep(1)
    
    pyautogui.press('f8')
    print('\n\nStart playing...')
    
    end_epoch = True
    
    errors  = 0
    
    salva = 0
    
    _time_ = 60
    
    epoch = 0
    while end_epoch:
        
        _time_ = return_time(queue[-1][0],time_model)
        
        print(_time_)
        
        if _time_ <= 10:
            
            print('Saving File...')
            
            save_to_file(queue, './ANALYSE_DATA/fila_treino_segundo_'+random_file_number+'_'+str(epoch)+'_position.pkl')
            
            epoch = epoch + 1
            
            print('File Saved.')
            #save_to_file(actions_analysis, './ANALYSE_DATA/fila_action_time_'+random_file_number+'.pkl')
            salva = 0
            
        time.sleep(1)