##################################
#           TIME MODEL           #
##################################

import numpy as np 
import tensorflow as tf 
import global_variables

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

    time = vec[:10].argmax()*10 + vec[10:].argmax()
    
    return time

def return_time(image,model):
    
    return vec_to_captcha(model.predict(np.expand_dims(np.expand_dims(image[7:23,56:73],axis=-1),axis=0))[0])
