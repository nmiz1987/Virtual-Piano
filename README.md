# Virtual-Piano

virtual piano was created as part of a graduation project for a degree in software engineering at Afeka College.
Many thanks to project facilitator Mr. Amit Shtekel and partners Kirill Bortman.

## Purpose of the project

to create a system that the player can use to simulate piano playing experience, without the need to carry a large and heavy piano, and without the need to purchase additional instruments and accessories beyond the existing camera in the cell phone.

## Main requirements

the ability to identify the fingertips position in each frame when the palm of the hand is in the frame.
Find the absolute position of the camera which will capture both palm of the hands in the specific angle to calculate the position of the fingertips both in X axis (what key was pressed) and Y axis (when was the key pressed) to find the exact moment of the keystroke and play the note.

## Implementation method

Create a video capture of the user playing the virtual piano in a defined period of time and creating a data file which contains information on each frame, afterwards the video capture and the data file are used to produce the output with the user’s keystrokes music sounds.
