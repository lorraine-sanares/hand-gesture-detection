#===================================================================================
# Name: KeyboardPanTilt.py
#-----------------------------------------------------------------------------------
# Purpose: Control a Pimoroni Pan-Tilt Hat using arrow keys and take photos with Pi Camera.
#===================================================================================

import curses
import time
import pantilthat
from picamera2 import Picamera2, Preview
from libcamera import Transform

# ==================== CAMERA SETUP ====================
camera = Picamera2()
camera.configure(camera.create_preview_configuration(
    main={"size": (1024, 768)},
    transform=Transform(hflip=True, vflip=True)  # Correct flipping
))
camera.start_preview(Preview.QTGL)
camera.start()

# ==================== CURSES SETUP ====================
screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)

# ==================== PAN-TILT SETUP ====================
pan = 0.0  # Horizontal movement
tilt = 0.0 # Vertical movement
pantilthat.pan(pan)
pantilthat.tilt(tilt)

deltaPan = 1.0
deltaTilt = 1.0
picNum = 1

try:
    while True:
        char = screen.getch()
        if char == ord('q'):
            break
        elif char == ord('p'):
            filename = f"image{picNum}.jpg"
            camera.capture_file(filename)
            screen.addstr(0, 0, f"Picture taken: {filename}     ")
            picNum += 1
        elif char == curses.KEY_RIGHT:
            if pan + deltaPan <= 90:
                pan += deltaPan
            pantilthat.pan(pan)
            screen.addstr(0, 0, f"Pan Right: {pan:.1f}     ")
        elif char == curses.KEY_LEFT:
            if pan - deltaPan >= -90:
                pan -= deltaPan
            pantilthat.pan(pan)
            screen.addstr(0, 0, f"Pan Left: {pan:.1f}      ")
        elif char == curses.KEY_DOWN:
            if tilt + deltaTilt <= 90:
                tilt += deltaTilt
            pantilthat.tilt(tilt)
            screen.addstr(0, 0, f"Tilt Down: {tilt:.1f}     ")
        elif char == curses.KEY_UP:
            if tilt - deltaTilt >= -90:
                tilt -= deltaTilt
            pantilthat.tilt(tilt)
            screen.addstr(0, 0, f"Tilt Up: {tilt:.1f}       ")
        screen.refresh()
        time.sleep(0.01)

finally:
    camera.stop_preview()
    camera.close()
    curses.nocbreak()
    screen.keypad(False)
    curses.echo()
    curses.endwin()
