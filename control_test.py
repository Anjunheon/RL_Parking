import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import pyautogui

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print("API Control enabled: %s\n" % client.isApiControlEnabled())
car_controls = airsim.CarControls()

pyautogui.click(1125, 455)
time.sleep(1)

client.reset()
time.sleep(1)

print("Go Forward")
car_controls.throttle = 1
client.setCarControls(car_controls)
time.sleep(3)

print("Apply Break")
car_controls.throttle = 0
car_controls.brake = 0.205
client.setCarControls(car_controls)

while client.getCarState().speed != 0:
    pass

# print("x :", round(client.getCarState().kinematics_estimated.position.x_val, 3))
# print("y :", round(client.getCarState().kinematics_estimated.position.y_val, 3))
print(client.getCarState())
print(client.getCarControls())
