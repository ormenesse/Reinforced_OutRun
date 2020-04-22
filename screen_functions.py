import subprocess
import pyautogui
import time

def generate_emulator_pid():

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

    upperleftx = int(output.split('\n')[3][-4:])
    upperlefty = int(output.split('\n')[4][-4:])

    # load quick save state
    
    pyautogui.press('f8')

    return p, upperleftx, upperlefty

def kill_process(p):

    p.kill()

def capture(graber,queue,positionx,positiony):

    while True:
    #print("Comecando a gavar...")
        sct_img = graber.grab({'left': positionx, 'top': positiony+40, 'width': 640, 'height': 446})
        Img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        Img.thumbnail((320,220), Image.ANTIALIAS)
        Img = np.array(Img.convert('L')).astype(np.uint8)
        Img = Img.T
        queue.enqueue(Img)
        time.sleep(0.15)