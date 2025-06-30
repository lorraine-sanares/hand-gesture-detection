#Raspberry-Pi pan and tilt using arrow keys script
#must be run from Pi's terminal!
#use code "python KeyboardPanTilt.py" after you cd into the correct folder!

#importing required libraries
import curses
import os
import time
import picamera
import pantilthat

# Initialise camera
camera = picamera.PiCamera()
camera.resolution = (1024, 768)
camera.start_preview(fullscreen=False, window = (100,20,640,480))

# flipping the camera for so its not upside down
camera.vflip = True
camera.hflip = True

# Set up key mappings and curses for arrow key responses
screen = curses.initscr() # get the curses screen window
curses.noecho()           # turn off input echoing
curses.cbreak()           # respond to keys immediately (don't wait for enter)
screen.keypad(True)       # map arrow keys to special values

# initialise pan and tilt positions and process increments driven by arrow keys
# set start up serrvo positions
a = 0.0
b = 0.0
pantilthat.pan(a)
pantilthat.tilt(b)
# set arrow key delta
deltaPan=1.0
deltaTilt=1.0
 
picNum = 1  # Initialise picture number

# Process active key presses:
# -- Letter p will take a picture and store file name image[picNum].jpg,
#     where [number] increments over a picture taking session.
# -- Arrow keys will control the Pan Tilt Camera (deltaPan/deltaTilt Degree angles)
# -- Letter q will quit the application, 
try:
    while True:
        char = screen.getch()
        if char == ord('q'):
            #if q is pressed quit
            break
        if char == ord('p'):
            #if p is pressed take a photo!
            camera.capture('image%s.jpg' % picNum)
            picNum = picNum   1
            screen.addstr(0, 0, 'picture taken! ')
        elif char == curses.KEY_RIGHT:
            screen.addstr(0, 0, 'right ')
            if (b - deltaTilt ) > -90:
                b = b - deltaTilt
            pantilthat.pan(b)
            time.sleep(0.005)
        elif char == curses.KEY_LEFT:
            screen.addstr(0, 0, 'left ')
            if (b   deltaTilt) < 90:
                b = b   deltaTilt
            pantilthat.pan(b)
            time.sleep(0.005)
        elif char == curses.KEY_DOWN:
            screen.addstr(0, 0, 'down ')
            if (a   deltaPan) < 90:
                a = a   deltaPan
            pantilthat.tilt(a) 
            time.sleep(0.005)
        elif char == curses.KEY_UP:
            screen.addstr(0, 0, 'up ')
            if (a - deltaPan) > -90:    
                a = a - deltaPan
            pantilthat.tilt(a)
            time.sleep(0.005)
                        
finally:
    # shut down cleanly
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()