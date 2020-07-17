##################################
#         DECISION MODEL         #
##################################

import global_variables
import numpy as np 
import tensorflow as tf 
import gc
import requests
import io
import zlib
import ast
from score_model import *
from functions import *
import time
import hashlib

################################
# GEAR CHANGE MODEL            #
# GEAR CHANGE MODEL            #
################################

def generate_gearchange_model():
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x/255,input_shape=(15, 76, 1)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=(15, 76, 1)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    model.load_weights('./GEARCHANGE_MODEL/'+'model_change_outrun.h5')
    
    return model

def return_return_gearchange_score(image,model):

    score = model.predict(np.expand_dims(np.expand_dims(image[25:40,64:140],axis=-1),axis=0))[0]
    
    return 1 if score > 0.6 else 0


################################
# ACTION ENCODED MODEL         #
# RIGHT LEFT ACC BRAKE         #
################################

# action model
def generate_model():

    json_file = open("./Action_Encoded_Model/modelo_outrun.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights("./Action_Encoded_Model/modelo_outrun.hdf5")
    print("Loaded model from disk")
    
    return model

def capture_return_decision(images):
    
    # predict command
    imgs = np.array(images).astype(np.uint8)
    imgs = np.expand_dims(imgs, axis=0)
    imgs = np.swapaxes(imgs,1,2)
    imgs = np.swapaxes(imgs,2,3)
    
    """
    try:
        print(hashlib.md5(imgs.tobytes()).hexdigest())
    except Exception as inst:
        print('erro hash')
        print(inst)
    """
    
    return global_variables.model.predict(imgs.astype(np.float32))

def model_capture_return_images(images): 
    
    # predict command
    imgs = np.array(images).astype(np.uint8)
    #imgs = np.expand_dims(imgs, axis=0)
    imgs = np.swapaxes(imgs,0,1)
    imgs = np.swapaxes(imgs,1,2)
    #print('model_capture_return_images shape',imgs.shape)
    return imgs.astype(np.float32)

################################
# SENDING DATA TO              #
#  SERVER                      #
################################

def compress_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)

def retrain_and_do_all_on_server(server_ip='192.168.132.1:5000'):
    
    global_variables.capture_screen = False
    time.sleep(0.5)
    print('Training will have: ',len(global_variables.queue),'samples over 5 epochs.')
    
    try:
        # Compressing Data
        compressed_data, u_sz, c_sz = compress_nparr(global_variables.queue)
    except:
        # Compressing Data
        compressed_data, u_sz, c_sz = compress_nparr(global_variables.queue[:-1])
    #seding request
    r = requests.post('http://'+server_ip, data=compressed_data, headers={'Content-Type': 'application/octet-stream'}, timeout=600)
    
    # freeing memory
    global_variables.queue = []
    
    try:
        info_text = ast.literal_eval(r.text)
    except:
        info_text = { 'Message' : 'Error.'}
    
    if info_text['Message'] == 'Model Trained.':
        # Downloading the new Trained Model
        url = 'http://'+server_ip+'/download'
        r = requests.get(url, allow_redirects=True)
        open('./Action_Encoded_Model/modelo_outrun.hdf5', 'wb').write(r.content)
        
    ################################
    # reloading model              #
    # compiled models are too slow #
    ################################
    global_variables.model = generate_model()
    
    global_variables.queue = []
    
    # restarting
    pyautogui.press('f8')
    
    global_variables.capture_screen = True
    
    # Cleaning Memory
    gc.collect()
    
    return True