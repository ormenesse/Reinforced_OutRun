##################################
#           SCORE MODEL          #
##################################

import numpy as np 
import tensorflow as tf
import gc
import zlib
import cv2

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
    
    model.load_weights('model_score_outrun.h5')
    
    return model

def return_score(image,model):
    limit_numbers_1 = np.arange(200,136,-8)
    limit_numbers_2 = np.arange(192,128,-8)
    
    score = 0
    
    for j in range(len(limit_numbers_1)):
        
        i = model.predict(np.expand_dims(np.expand_dims(image[15:24,limit_numbers_2[j]:limit_numbers_1[j]],axis=-1),axis=0))[0]
        
        score = score + i.argmax()*(10**j)
    
    return score


def generate_position_model():
    input_shape =  (80, 200, 1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x/255,input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='linear'))
    
    model.load_weights('model_position_score_outrun.h5')
    
    return model

def return_position_score(image,model):
        
    score = model.predict(np.expand_dims(np.expand_dims(process_img(image[140:,50:250]),axis=-1),axis=0))[0]
    
    return score

def generate_velocity_model():
    
    input_shape =  (28, 16, 1)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x/255,input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    
    model.load_weights('model_velocity_outrun.h5')
    
    return model

def return_velocity_score(image,model):
        
    limit_numbers_1 = np.arange(33,9,-8)
    limit_numbers_2 = np.arange(41,17,-8)
    
    img = process_img( cv2.resize(image, (640,440), interpolation = cv2.INTER_AREA))
    
    velocity = 0
    
    for j in range(len(limit_numbers_1)):
        
        i = model.predict(np.expand_dims(np.expand_dims(img[25*2:39*2,limit_numbers_1[j]*2:limit_numbers_2[j]*2],axis=-1),axis=0))[0]
        
        velocity = velocity + i.argmax()*(10**(j))
    
    return velocity

#
# REINFORCEMENT LEARNING MODEL
#

def capture_return_decision(model,images):
    
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
    
    return model.predict(imgs.astype(np.float32))

def model_capture_return_images(images): 
    
    # predict command
    imgs = np.array(images).astype(np.uint8)
    #imgs = np.expand_dims(imgs, axis=0)
    imgs = np.swapaxes(imgs,0,1)
    imgs = np.swapaxes(imgs,1,2)
    #print('model_capture_return_images shape',imgs.shape)
    return imgs.astype(np.float32)

# action model
def generate_model():

    json_file = open("./modelo_outrun_ae.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights("./modelo_outrun_ae.hdf5")
    print("Loaded model from disk")
    
    return model


def process_img(image, sigma=0.4):
	# compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def retrain_model(queue,epoch=1):
    
    #epoch yet to be implemented.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    score_model = generate_scoring_model()
    position_score_model = generate_position_model()
    velocity_model = generate_velocity_model()

    #Applying Q-Learning
    # learning rate - in reinforcement learning, this is gamma
    """
    LR = 0.5 * (1 - ((epoch-1)/10000))

    if LR < 0.1:
        LR = 0.1
    """

    LR = 0.05

    actionList = []
    act_rightleft = []
    frame_imgs = []
    score = []
    score_accbrake = []
    score_position = []
    velocity = []
    velocityModel = []
    time = []

    print('Starting to organize data before training... Nº', 0, 'of', len(queue),'samples.                ', end='\r')
    for i in np.arange(5,len(queue)-5,1):
        print('Starting to organize data before training... Nº', i, 'of', len(queue),'samples.             ', end='\r')
        frame_imgs.append(model_capture_return_images( [ process_img(j[0][120:,60:270]) for j in queue[i-4:i+1]]))
        # model actions
        actionList.append(queue[i][1][0])
        # appending score
        score_i = return_score(queue[i][0],score_model)
        score_f = return_score(queue[i+5][0],score_model)

        score_points = np.clip( ( score_f - score_i)/(5*700) ,0,1)

        score_pos = return_position_score(queue[i+1][0],position_score_model)[0]

        vel = np.clip( return_velocity_score(queue[i][0],velocity_model)/170, 0,1)

        velocity.append(vel)

        velocityModel.append(np.clip( return_velocity_score(queue[i+5][0],velocity_model)/170, 0,1) - vel)


        if vel == 0 and score_points == 0:

            score_accbrake.append(0)

            score_position.append(0)

            score.append( np.log(1) )

        else:

            score_accbrake.append( score_points )

            #score_position.append( np.clip( score_pos[0] + np.clip((score_points/1000) if score_points > 100 else -0.5, -0.5, 0.1) , -0.5, 1) )
            score_position.append( score_pos) 

            score.append( np.log( np.clip(velocityModel[-1]/2 + score_pos*2 + score_points/2, 0.0001, 100) ) )

        #time between samples
        time.append(queue[i+5][2]-queue[i][2])

        # Q - Learning Happening Here

        action = actionList[-1].copy()

        for i,j in enumerate(action):

            if i == action.argmax():

                action[i] = np.clip(action[i] + LR*score[-1],0,1)

            else:

                action[i] = np.clip(action[i],0,1)
                #action[i] = np.clip(action[i] - LR*score[-1],0,1)

        actionList[-1] = action.copy()

        ###                           ###

        # Just some causality for idiotic future
        if vel == 0 and score_points == 0:

            for p in range(i,i-15,-1):
                try:
                    
                    if score_position[p] > 0.7:
                        
                        break
                        
                    else:
                    
                        actionList[p][actionList[p].argmax()] = np.clip(actionList[p][actionList[p].argmax()] - 0.1, 0, 1)

                        score[p] = np.log( np.clip(velocityModel[p]/2 + score_position[p]*2 + score_accbrake[p]/2 - 0.5, 0.0001, 100) ) 

                except:
                    pass

    # pre trained model
    # create train 
    dict_train = {}
    dict_train['output'] = np.array(actionList)

    frame_imgs = np.array(frame_imgs)

    # the only way it works
    model = generate_model()
    model.load_weights("D:/TensorflowAPI/modelo_outrun_ae.hdf5")
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mse','mae'])

    model.fit(frame_imgs, dict_train,
              batch_size=20,
              epochs=1,
              shuffle = True,
              validation_data=(frame_imgs, dict_train)
             )

    model.save_weights("D:/TensorflowAPI/modelo_outrun_ae.hdf5")
    
    print('AccBrake',np.round(dict_train['output'][-20:]*100),2)
    
    with open("TrainOutput.txt", "a") as output:
        output.write('\n')
        output.write('Learning Rate: '+str(LR)+'.\n')
        output.write('Sample has ' + str(len(queue)) + ' training samples.\n')
        output.write('Mean time between samples: '+ str(np.mean(time)) + ' sec.\n')
        output.write('Mean Score Points between samples: ' + str(np.mean(score_accbrake)) + ' points.\n')
        output.write('Maximum Delta Score Points:' + str(max(score_accbrake)) + '.\nMinimum Delta Score Points:' + str(min(score_accbrake))+'.\n')
        output.write('Mean Score (Right & Left) between samples: ' + str(np.mean(score_position)) + ' points.\n')
        output.write('Maximum Delta Score (Right & Left):' + str(max(score_position)) + '.\nMinimum Delta Score (Right & Left):' + str(min(score_position))+'.\n')
        output.write('Mean Score Velocity between samples: ' + str(np.mean(velocityModel)) + ' points.\n')
        output.write('Maximum Delta Score Velocity:' + str(max(velocityModel)) + '.\nMinimum Delta Score Velocity:' + str(min(velocityModel))+'.\n')
        output.write('Maximum Train Score: ' + str(return_score(queue[-1][0],score_model))+'.\n')
        output.write('Mean Velocity (MPH):' + str(np.mean(velocity)*170))
        output.write('\n')
    
    output.close()
    
    # Cleaning Memory
    gc.collect()
    
    return True