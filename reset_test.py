import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import tempfile
import pyautogui
import pytesseract
from PIL import Image

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
    client.enableApiControl(False)
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


def capture_goal():
    # 언리얼에서 출력되는 목표 지점 좌표
    unreal_goals = [[600, 2600], [600, 2230], [600, 1800], [600, 1430], [600, 990], [600, 620],  # 우측
                    [-1200, 2600], [-1200, 2230], [-1200, 1800], [-1200, 1430], [-1200, 990]]  # 좌측

    # 에어심 API를 통해 출력되는 목표 지점 좌표
    airsim_goals = [[6, -14], [6, -17], [6, -22], [6, -25], [6, -30], [6, -33],  # 우측
                    [-7, -14], [-7, -17], [-7, -22], [-7, -25], [-7, -30]]  # 좌측

    # 좌표 출력 부분 스크린샷 캡쳐
    img = pyautogui.screenshot('goal.png', region=(36, 90, 210, 15))  # 전체화면(F11) 기준
    # 좌표 스크린샷 문자열로 변환
    goal_pos = pytesseract.image_to_string(Image.open('goal.png'))
    # print(goal_pos[:-2])

    goal_pos = str.split(goal_pos[:-2], ' ')

    x = str.split(goal_pos[0], '.')[0]
    y = str.split(goal_pos[1], '.')[0]
    x = int(float(x[2:]))
    if y[0] == '¥':
        y = int(float(y[3:]))
    else:
        y = int(float(y[2:]))

    goal_xy = []
    for i in range(len(airsim_goals)):
        if x == unreal_goals[i][0] and y == unreal_goals[i][1]:
            # print('Goal x :', airsim_goals[i][0])
            # print('Goal y :', airsim_goals[i][1])
            goal_xy = airsim_goals[i]
            print('Goal :', airsim_goals[i])
            break

    return goal_xy


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


# 처음 실행 시 충돌 물체 인식 후 리셋 관련 문제 때문에 중지 후 다시 시작
client, car_controls = sim_start()
sim_stop()
client, car_controls = sim_start()

time.sleep(2)

fin = 0  # 목적지 좌표 출력 flag
while True:
    if fin == 0:
        goal = capture_goal()
        fin = 1
        # car_controls.throttle = 1
        # car_controls.steering = 1
        # client.setCarControls(car_controls)

    # print((client.simGetCollisionInfo().object_name).lower())
    print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b', end='')
    print('x_val :', round(client.getCarState().kinematics_estimated.position.x_val, 3),
          'y_val :', round(client.getCarState().kinematics_estimated.position.y_val, 3), end='')
    if (goal[0] > 0):
        if (client.getCarState().kinematics_estimated.position.x_val > 6 and
                client.getCarState().kinematics_estimated.position.x_val < 8 and
                client.getCarState().kinematics_estimated.position.y_val > goal[1] - 1 and
                client.getCarState().kinematics_estimated.position.y_val < goal[1] + 1):
            print("\nFINISH!\n")
            sim_stop()
            client, car_controls = sim_start()
            fin = 0
    elif (goal[0] < 0):
        if (client.getCarState().kinematics_estimated.position.x_val > -9 and
                client.getCarState().kinematics_estimated.position.x_val < -7 and
                client.getCarState().kinematics_estimated.position.y_val > goal[1] - 1 and
                client.getCarState().kinematics_estimated.position.y_val < goal[1] + 1):
            print("\nFINISH!\n")
            sim_stop()
            client, car_controls = sim_start()
            fin = 0

    if ((client.simGetCollisionInfo().object_name).lower()).find('pipesmall') >= 0:
        continue
    else:
        print("\nCRASH!!\n")
        sim_stop()
        client, car_controls = sim_start()
        fin = 0

    # time.sleep(1)
