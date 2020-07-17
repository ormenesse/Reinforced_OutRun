#https://stackoverflow.com/questions/24892035/how-can-i-get-the-named-parameters-from-a-url-using-flask
from flask import Flask, request, send_file, redirect, url_for, flash, jsonify, abort
from datetime import datetime
from flask_cors import CORS
from models import *
import tensorflow as tf
import numpy as np
import json
import zlib
import gc
import io
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app) #Prevents CORS errors

epoch_counter = 1

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

@app.route('/', methods=['POST'])
def works():
    
    global epoch_counter
    
    gc.collect()
    
    print('requisicao recebida...')
    
    r = request
    #
    data = uncompress_nparr(r.data)
    
    print('Retraining Model...')
    
    process_eval = multiprocessing.Process(target=retrain_model, args=(data,epoch_counter))
    process_eval.start()
    process_eval.join()
    
    epoch_counter = epoch_counter + 1
    
    """
    try:
        answer = retrain_model(data,1)
    except:
        print('Train Error')
        answer = False
    """
    
    print(process_eval)
    print('Ending...')
    
    return jsonify({'Message':'Model Trained.'})

@app.route('/download')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "D:/TensorflowAPI/modelo_outrun_ae.hdf5"
    return send_file(path, as_attachment=True)
    

if __name__ == "__main__":
    
    app.run(host="0.0.0.0")