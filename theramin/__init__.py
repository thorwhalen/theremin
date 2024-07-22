"""

The objective of this project is to make "instruments" that work like a theramin, 
that is, whereby the features of the sounds produced are controlled by the position of 
the left and right hand. 

That, but generalized. 
A theramin is an analog device that whose sound is parametrized by two features only:
Pitch and volume, respectively controlled by the left and right hand; the 
distance from the device's sensor determines the pitch and volume of the sound produced.

Here, though, we'll use video as our sensor, so we can not only detect the positions 
of the hands, but also the positions of the fingers, and the angles between them, 
detect gestures, and so on. Further, we can use the facial expressions to add to 
the parametrization of the sound produced.

What the project wants to grow up to be is a platform for creating musical instruments
that are controlled by the body, enabling the user to determine the mapping between
the video stream and the sound produced. 
We hope this will enable the creation of new musical instruments that are more
intuitive and expressive than the traditional theramin.

Right now, the project is very much in its infancy, and we're just trying to get the
basics working.

Here's a bit about what exists so far:

* _03_hand_gesture_recognition_with_video_feedback.py: Detects hand gestures and overlays
    the detected landmarks and categories on the video stream in real-time.
* _06_hand_gesture_w_slabs.py: The same, but using slabs, for a better framework setup
* _05_a_wrist_lines.py: Draws lines showing the wrist position on the video stream
    in real-time.
* _07_wrist_lines_w_sound.py: Draws lines showing the wrist position on the video stream
    and produces sound based on the wrist position. The resulting sound is horrible though, 
    due to the lack of a proper real-time setup (I believe). The sound produced is 
    produced in realtime, but only a small waveform chunk is produced. 
    It's choppy.
    **We need to buffer the stream of sound features and have an independent process that
    reads from the buffer and produces the sound, taking more than one chunk of 
    information to determine the sound produced.**
* _08_keyboard_theramin.py: To reduce the complexity of the sound produced, this script
    uses a keyboard to produce sound, with the pitch and volume controlled by the 
    left and right hand, respectively. This doesn't work well at all. More work is needed.



"""