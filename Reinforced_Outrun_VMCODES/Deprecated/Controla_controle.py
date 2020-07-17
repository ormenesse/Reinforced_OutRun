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
    
        #controller.
    monitor_thread = threading.Thread(target=replicate, args=(dev, virtual_gamepad))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    while True:
        pass