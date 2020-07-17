#!/usr/bin/env python3
# Always add this line on the begginig 
# #chmod +0777 /dev/uinput

# ../../anaconda3/bin/python NetworkTrain_Outrun.py -i 192.168.132.1:5000 -s false -r false

###########################
# LOADING EVDEV LIBRARIES #
###########################
import asyncio
from evdev import list_devices, InputDevice, InputEvent, categorize, ecodes, UInput
import evdev
import subprocess
import pyautogui
import time

##################################
# LOADING              LIBRARIES #
##################################

import threading
import numpy as np
from PIL import Image
import mss
import sched, time
import cv2
import pickle
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import subprocess
import os, sys, getopt
import re 
import gc
import requests
import io
import zlib
import ast

##################################
# LOADING DEEPLEARNING LIBRARIES #
##################################
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

##################################
# LOADING FUNCTIONS & CLASSES    #
##################################

from functions import *
from score_model import *
from decision_model_functions import *
from gameover_model import *
        
##################################
#              MAIN              #
##################################        

import global_variables

global_variables.init()

if __name__ == '__main__':
    
    print('First of all is to use the command as following:\n\n#chmod +0777 /dev/uinput\n\n\n\n')
    
    """
    ARGV`S OPTIONS
    """
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:s:r:")
    except getopt.GetoptError:
        print('Usage: train.py -i <IP:PORT> -s <save_queue_epoch=true|false> -r <playRandom=true|false>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -i <IP:PORT> -s <save_queue_epoch=true|false> -r <playRandom=true|false>')
            sys.exit()
        elif opt in ("-i"):
            server_ip = arg
        elif opt in ("-s"):
            if arg == 'true':
                save_queue = True
            else:
                save_queue = False
        elif opt in ("-r"):
            if arg == 'true':
                global_variables.randomplay = True
            else:
                global_variables.randomplay = False
    
    """
    HOOK GAMEPAD.
    """
    # Listen to events from the following device.
    print('Will Inject on device:',list_devices()[0])
    dev = InputDevice( list_devices()[0] )
    print('Device Hooked.')
    # Then create our own devices for our own events.
    # (It is possible to write events directly to the original device, but that has caused me mysterious problems in the past.)
    global_variables.virtual_gamepad = UInput.from_device(dev, name="Virtual Microsoft X-Box 360 pad")#name="virtual-gamepad")

    # Now we monopolize the original devices.
    dev.grab()

    # Sends all events from the source to the target, to make our virtual keyboard behave as the original.
    


    print('Virtual Gamepad created. \nReady to play videogames!')
    
    """
        LOADING MODEL
    """
    
    print('\n\nLoading Models...')
    global_variables.score_model = generate_scoring_model()
    global_variables.time_model = generate_time_model()
    global_variables.model = generate_model()
    global_variables.position_score_model = generate_position_model()
    global_variables.gear_model = generate_gearchange_model()
    
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
    monitor_thread = threading.Thread(target=replicate, args=(dev, global_variables.virtual_gamepad))
    #monitor_thread.daemon = True
    monitor_thread.start()
    
    # Apply Commands Thread
    #apply_cmds_thread = threading.Thread(target=apply_model_controller_output)
    #monitor_thread.daemon = True
    #apply_cmds_thread.start()
    
    
    print('Waiting all modules to initialize...')
    
    pyautogui.press('f8')
    global_variables.capture_screen = True
    time.sleep(5)
    print('\n\nComputer in starting to play...')
    
    #####################
    # control variables #
    #####################
    global_variables.end_epoch = 0
    random_file_number = str(np.round(np.random.random()*1000,0))
    print('\n\n\nRandom File Number Generated:', random_file_number,'\n\n\n')
    
    # starting limit time in OutRun
    _time_ = 80
    
    while True:
        
        try:
            _time_ = return_time(global_variables.queue[-1][0],global_variables.time_model)
        except:
            _time_ = 80
        
        #print(global_variables.commands)
        
        if _time_ <= 1: # Finding game over not yet implemented.
            
            global_variables.end_epoch = global_variables.end_epoch + 1
            if save_queue:
                print('Saving File...')
                save_to_file(global_variables.queue, './ANALYSE_DATA/fila_treino_'+random_file_number+'_'+str(global_variables.end_epoch)+'.pkl')
            #
            print('Analysing Epoch...')
            retrain_and_do_all_on_server(server_ip)
            gc.collect()
            
        time.sleep(0.3)