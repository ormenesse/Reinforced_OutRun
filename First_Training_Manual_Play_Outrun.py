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
    
    #defining global variables
    global queue
    global commands
    global controller
    
    while True:
    #print("Comecando a gavar...")
        sct_img = graber.grab({'left': positionx, 'top': positiony+40, 'width': 640, 'height': 440})
        Img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        Img.thumbnail((320,220), Image.ANTIALIAS)
        Img = np.array(Img.convert('L')).astype(np.uint8)
        
        score = return_score_cv_result(Img)
        
        comandos = controller.read()
        
        direita = [1 if comandos[0] == 1 else 0][0]
        esquerda = [1 if comandos[0] == -1 else 0][0]
        #baixo = comandos[1].map(lambda x: 1 if x == 1 else 0)
        #cima = comandos[1].map(lambda x: 1 if x == -1 else 0)
        
        #           accelerate    brake        right      left       gear up&down
        commands = [ comandos[2], comandos[3], direita , esquerda , comandos[4] ]
        
        queue.append((Img.copy(),commands.copy(),score))
        
        time.sleep(0.2)
    
def replicate(source_device, target_device):
    for event in source_device.async_read_loop():
        #print('Man Generated Event',event)
        target_device.write_event(event)

        
##################################
#           SCORE MODEL          #
##################################

# using OpenCV
import cv2, pytesseract
def return_score_cv_result(img):
    res = cv2.resize(img[14:25,135:201], dsize=(4*img[14:25,135:201].shape[1],4*img[14:25,135:201].shape[0]), interpolation=cv2.INTER_CUBIC)
    rest = pytesseract.image_to_string(cv2.blur(res,(5,5)),nice=1,config='--psm 7 --oem 3 digits')
    rest = rest.replace('o','0')
    #print('Pytesseract leu:', rest)
    try:
        return int(rest)
    except:
        return 0

##################################
#              MAIN              #
##################################        

# global Queue
queue = []

#global command variable
commands = []

# global read XboxController
controller = XboxController()

if __name__ == '__main__':
    
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
    
    
    print('Waiting all modules to initialize...')
    time.sleep(1)
    
    pyautogui.press('f8')
    print('\n\nStart playing...')
    
    end_epoch = True
    
    errors  = 0
    
    salva = 0
    
    while end_epoch:
        
        salva = salva + 1
        
        try:
            # control it is working
            print(str(commands))
            
        except Exception as e: 
            
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
            errors = errors + 1
        
        if salva == 195:
            
            print("Salvando em menos de 1 segundo;")
        
        elif salva == 200:
            
            print('Salvando arquivo.')
            
            save_to_file(queue, './Analyse_Data/fila_primeiro_treino.pkl')
            
            salva = 0
        
        if errors == 5:
            kill_process(p)
            break
            
        time.sleep(0.2)
