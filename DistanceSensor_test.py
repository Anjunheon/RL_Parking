import setup_path
import airsim
import cv2
import numpy as np
import os
import time

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
print("API Control enabled: %s\n" % client.isApiControlEnabled())
car_controls = airsim.CarControls()

while True:
    print(client.getDistanceSensorData("Distance1").distance)
    print(client.getDistanceSensorData("Distance2").distance)
    print(client.getDistanceSensorData("Distance3").distance)
    print(client.getDistanceSensorData("Distance4").distance)

    time.sleep(1.5)