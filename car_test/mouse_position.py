import time
import pyautogui
import pytesseract
from PIL import Image

while True:
    print(pyautogui.position())
    time.sleep(1)  # (40, 110) (240, 123)

    # 좌표 출력 부분 스크린샷 캡쳐
    img = pyautogui.screenshot('goal.png', region=(36, 90, 210, 15))  # 전체화면(F11) 기준
    # 좌표 스크린샷 문자열로 변환
    goal_pos = pytesseract.image_to_string(Image.open('goal.png'))
    # print(goal_pos[:-2])

    # x, y 좌표 구분 -> 좌표 값 float 변환
    goal_pos = str.split(goal_pos[:-2], ' ')

    x = str.split(goal_pos[0], '.')[0]
    y = str.split(goal_pos[1], '.')[0]

    x = int(float(x[2:]))
    if y[0] == '¥':  # 가끔 문자를 잘못 인식하는 경우 발생
        y = int(float(y[3:]))
    else:
        y = int(float(y[2:]))