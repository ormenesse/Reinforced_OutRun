##################################
# LOADING FUNCTIONS & CLASSES    #
##################################

import asyncio
import evdev
from evdev import list_devices, InputDevice, InputEvent, categorize, ecodes, UInput
import evdev
import subprocess
import pyautogui
import time
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
import os, sys
import re 
import gc
import requests
import io
import zlib
import ast

#

from decision_model_functions import *
import global_variables

#

def save_to_file(objeto, nome_arquivo):
    with open(nome_arquivo, 'wb') as output:
        pickle.dump(objeto, output, pickle.HIGHEST_PROTOCOL)

def load_file(nome_arquivo):
    with open(nome_arquivo, 'rb') as input:
        objeto = pickle.load(input)
    return objeto

def demote(user_uid, user_gid):
    """Pass the function 'set_ids' to preexec_fn, rather than just calling
    setuid and setgid. This will change the ids for that subprocess only"""

    def set_ids():
        os.setgid(user_gid)
        os.setuid(user_uid)

    return set_ids

def generate_emulator_pid():

    # loading game in emulator
    p = subprocess.Popen(['gens','/home/ormenesse/Documents/Reinforced_Outrun/OutRun (USA, Europe).md'])
    # wait for the process to start
    time.sleep(5)

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
    
def process_img(image, sigma=0.4):
	# compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def capture(graber,positionx,positiony):
    
    #defining global variables
    """
    global queue
    global commands
    global model 
    global capture_screen
    global end_epoch
    global randomplay 
    global randomcommand
    global randomb
    """
    #Initializing Random
    
    count_capture = 0
    repeat_randomplay = 0
    notChangedGear = True
    frames_to_random = 20
    
    while True:
        
        if global_variables.capture_screen:
            
            time_1 = time.time()
            
            sct_img = graber.grab({'left': positionx, 'top': positiony+40, 'width': 640, 'height': 440})
            Img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
            Img.thumbnail((320,220), Image.ANTIALIAS)
            Img = np.array(Img.convert('L')).astype(np.uint8)
            
            # CHECK GEAR CHANGE
            if notChangedGear:
                gearchange = return_return_gearchange_score(Img.copy(),global_variables.gear_model)
                notChangedGear = False
            else:
                gearchange = 0
                notChangedGear = True
            
            # THE FIRST 2 IF`S ARE FOR RANDOMPLAY WHILE TRAINING THE NETWORK
            
            if count_capture == (frames_to_random+global_variables.end_epoch) and global_variables.randomplay == True and repeat_randomplay == 0:
                
                global_variables.randomcommand, global_variables.randomb = get_controller_action_random(gearchange)
                
                global_variables.commands = global_variables.randomcommand.copy()
                
                b  = global_variables.randomb.copy()
                
                repeat_randomplay = repeat_randomplay + 1
                
                #print('Random!', global_variables.commands)
            
            elif count_capture == (frames_to_random+global_variables.end_epoch) and global_variables.randomplay == True and repeat_randomplay < 5:
                
                global_variables.commands = global_variables.randomcommand.copy()
                
                global_variables.commands[4] = gearchange
                
                b  = global_variables.randomb.copy()
                
                count_capture = 0
                
                repeat_randomplay = 0

            else:
                
                try:
                    decision_array = [ j[0] for j in global_variables.queue[-4:].copy()]
                    decision_array.append(Img.copy())
                    b = capture_return_decision([ process_img(j[120:,60:270]) for j in decision_array])
                    #print(np.round(b,2))
                    global_variables.commands = get_controller_action_model(b,gearchange)
                    
                except:
                    
                    global_variables.commands = [0, 0, 0, 0, 0]
                    
                    b = np.zeros((1,12))

                count_capture = count_capture + 1

            # Add image do Queue
            global_variables.queue.append((Img.copy(),b[0].copy(),time.time()))
            
            #wrinting to controller output
            write_to_controller_output(global_variables.commands.copy())
            
            time_2 = time.time()
        
            if time_2 - time_1 < 0.2:

                time.sleep(0.2-(time_2 - time_1))
    
def replicate(source_device, target_device):
    for event in source_device.async_read_loop():
        #print('Man Generated Event',event)
        target_device.write_event(event)
        
def apply_model_controller_output():

    while True:
        if global_variables.capture_screen:
            write_to_controller_output(global_variables.commands)
        
def write_to_controller_output(command):
    #global virtual_gamepad
    # [accelerate    brake        right      left       gearup&down]
    
    # Gear
    if command[4] == 1:
        global_variables.virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_EAST, 1)
        
    else:
        global_variables.virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_EAST, 0)
    
    #Acc and Brak
    if command[0] == 1 and command[1] == 0:  # accelerate 1 and brake 0
        global_variables.virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 1)
        global_variables.virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 0)
    elif command[1] == 1 and command[0] == 0: # accelerate 0 and brake 0
        global_variables.virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 0)
        global_variables.virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 1)
    else: # else do nothing
        global_variables.virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 0)
        global_variables.virtual_gamepad.write(ecodes.EV_KEY, ecodes.BTN_NORTH, 0)

    # left righ decision
    direcionalx = 0 
    if command[2] == 1 and command[3] == 0:
        direcionalx = 1 # go right
    elif command[2] == 0 and command[3] == 1:
        direcionalx = -1 # go left
    else: # go straight ahead
        direcionalx = 0

    #working with directionals in the controller.
    global_variables.virtual_gamepad.write(ecodes.EV_ABS,ecodes.ABS_HAT0X,direcionalx)
    #virtual_gamepad.write(ecodes.EV_ABS,ecodes.ABS_HAT0Y,direcionaly)
    global_variables.virtual_gamepad.syn()


def get_controller_action_model(predict,gear):
    
    # plausible positions I got when gaming
    # [accelerate    brake        right      left       gearup&down]
    # [10000*i[0] +  1000*i[1]    +100*i[2]  +10*i[3]     +1*i[4]    for i in cmds]
    cmds = [0, 1, 10, 100, 101, 110, 1000, 1001, 1010, 1100, 1101, 1110]
    
    argument = predict.argmax()
    
    cmd_number = cmds[argument]
    
    if cmd_number == 0:
        
        return [0, 0, 0, 0, 0]
    
    else:
        
        acelera = cmd_number // 10**3 % 10
        
        freia = cmd_number // 10**2 % 10
        
        direita = cmd_number // 10**1 % 10
        
        esquerda = cmd_number // 10**0 % 10
        
        #gear = cmd_number // 10**0 % 10
        
        return [acelera, freia, direita, esquerda, gear]
    
def get_controller_action_random(gear):
    
    # plausible positions I got when gaming
    # [accelerate    brake        right      left       gearup&down]
    # [10000*i[0] +  1000*i[1]    +100*i[2]  +10*i[3]     +1*i[4]    for i in cmds]
    # 0	1	10	100	101	110	1000	1001	1010	1100	1101	1110
    cmds = [0, 1, 10, 100, 101, 110, 1000, 1001, 1010, 1100, 1101, 1110]
    
    cmd_number = np.random.choice(cmds,1)[0]
    
    b = np.zeros((1,12))
    b[:,cmds.index(cmd_number)] = 1
    
    if cmd_number == 0:
        
        return [0, 0, 0, 0, 0], b
    
    else:
        
        acelera = cmd_number // 10**4 % 10
        
        freia = cmd_number // 10**3 % 10
        
        direita = cmd_number // 10**2 % 10
        
        esquerda = cmd_number // 10**1 % 10
        
        #gear = cmd_number // 10**0 % 10
        
        return [acelera, freia, direita, esquerda, gear], b