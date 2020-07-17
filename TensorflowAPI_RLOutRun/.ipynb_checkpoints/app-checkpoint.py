#https://stackoverflow.com/questions/24892035/how-can-i-get-the-named-parameters-from-a-url-using-flask
from flask import Flask, request, send_file, redirect, url_for, flash, jsonify, abort
from datetime import datetime
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import zlib
import gc
import io

app = Flask(__name__)
CORS(app) #Prevents CORS errors


# ## HELPERS

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

def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)),allow_pickle=True)

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


def retrain_model(frame_imgs,cmds):
    
    model = generate_model()
    # pre trained model
    model.load_weights('./modelo_outrun.hdf5')
    
    frame_imgs = np.array(frame_imgs)
    
    #Train model
    INIT_LR = 1e-3
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer='adam',
                  metrics=['mse'])
    
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,verbose=0, mode='min')

    model.fit(frame_imgs[:,:,:,:], cmds,
              batch_size=32,
              epochs=10,
              shuffle = True,
              validation_data=(frame_imgs[:,:,:,:], cmds),
              callbacks=[earlyStopping])
    test_score_loss = model.evaluate(frame_imgs[:,:,:,:], cmds, verbose=0)
    
    #model.save("./Action_Encoded_Model/modelo_outrun.hdf5")
    model.save_weights("./modelo_outrun.hdf5")
    
    del model,frame_imgs,cmds
    # Cleaning Memory
    gc.collect()
    
    return True


@app.route('/', methods=['POST'])
def works():
    
    print('requisicao recebida')
    
    r = request
    #
    data = uncompress_nparr(r.data)
    
    print(type(data))
    
    print(data.shape)
    
    print(np.array(data[0,:].tolist()).shape)
    print(np.array(data[1,:].tolist()).shape)
    
    print('Retraining Model')
    
    retrain_model(np.array(data[0,:].tolist()),np.array(data[1,:].tolist()))
    
    return jsonify({'Message':'Model Trained.'})

@app.route('/download')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "modelo_outrun.hdf5"
    return send_file(path, as_attachment=True)
    

if __name__ == "__main__":
    
    app.run(host="0.0.0.0")