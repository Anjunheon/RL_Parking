import setup_path
import cv2 as cv
import numpy as np
import airsim
import time
import pyautogui

api_control = False

def sim_start():  # 시뮬레이터 실행
    # print(pyautogui.position())  # (1125, 455)
    pyautogui.click(1125, 455)
    # time.sleep(1)

    pyautogui.keyDown('altleft')
    pyautogui.keyDown('p')
    pyautogui.keyUp('altleft')
    pyautogui.keyUp('p')
    time.sleep(1)

    pyautogui.click(1125, 455)

    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(api_control)
    print("API Control enabled: %s\n" % client.isApiControlEnabled())
    car_controls = airsim.CarControls()

    time.sleep(1)

    return client, car_controls


def sim_stop():  # 시뮬레이터 중지
    # print(pyautogui.position())  # (1125, 455)
    pyautogui.click(1125, 455)
    time.sleep(1)

    # 시뮬레이터 종료
    pyautogui.keyDown('esc')
    pyautogui.keyUp('esc')
    time.sleep(1)


mapimg = cv.imread('map.png', cv.IMREAD_GRAYSCALE)
img_y, img_x = np.shape(mapimg)

x_len = 22
y_len = 45

x_block = round(np.shape(mapimg)[1]/x_len, 1)
y_block = round(np.shape(mapimg)[0]/y_len, 1)

# client, car_controls = sim_start()

cnt = 0
while cnt < 5:
    mapimg = cv.imread('map.png', cv.IMREAD_GRAYSCALE)

    client, car_controls = sim_start()

    s_t = time.time()
    e_t = time.time()

    while e_t - s_t < 15:
        x = round(client.getCarState().kinematics_estimated.position.x_val, 1) + 11
        y = round(client.getCarState().kinematics_estimated.position.y_val, 1) + 41

        x = int(x * x_block)
        y = int(y * y_block)

        x = 2 if x-2 < 0 else x
        x = img_x-3 if x+2 > img_x-3 else x
        y = 2 if y-2 < 0 else y
        y = img_y-3 if y+2 > img_y-3 else y

        mapimg[y-2:y+2, x-2:x+2] = 255

        e_t = time.time()

    cv.imwrite('.\\tracking\\tracking' + str(cnt) + '.png', mapimg)

    cnt += 1
    sim_stop()
    sim_stop()


