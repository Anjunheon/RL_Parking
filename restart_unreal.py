import time
import pyautogui


def err_restart():
    time.sleep(5)
    pyautogui.moveTo(960, 540)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    pyautogui.press('enter')
    time.sleep(30)

    pyautogui.press('win')
    time.sleep(3)
    pyautogui.press('u')
    pyautogui.press('n')
    pyautogui.press('r')
    pyautogui.press('e')
    pyautogui.press('a')
    pyautogui.press('l')
    pyautogui.press(' ')
    pyautogui.press('e')
    pyautogui.press('n')
    time.sleep(3)

    pyautogui.press('enter')
    time.sleep(20)

    pyautogui.moveTo(1030, 300)
    time.sleep(5)
    pyautogui.moveTo(500, 330)
    pyautogui.doubleClick()
    pyautogui.doubleClick()
    time.sleep(20)

    pyautogui.click(1030, 300)
    pyautogui.press('f11')
    time.sleep(20)
