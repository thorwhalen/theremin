# theremin

A video-to-sound theremin. 

To install:	```pip install theremin```


# About this project

The theremin, an iconic electronic musical instrument invented in the 1920s by [Léon Theremin](https://www.youtube.com/watch?v=w5qf9O6c20o), is played without physical contact, using hand movements to control pitch and volume through electromagnetic fields. 

Today, with advancements in digital sensors, AI-based models for real-time sensor data processing, and sophisticated audio generation tools, there is an unprecedented opportunity to generalize the theremin’s concept into a flexible framework and platform. This initiative, the theremin Project, aims to explore the sonification of physical activity, opening new avenues for musical expression.

Musical instruments are fundamentally devices that translate physical interactions into sound. The physical properties of an instrument—be it the strings of a violin or the keys of a piano—not only shape the sound produced but also influence how musicians interact with them. These interactions impose constraints that, far from being restrictive, inspire and guide creativity in profound ways. The theremin Project leverages modern digital technologies to expand this paradigm, enabling users to explore countless ways to produce sound through movement. By mapping visual cues (e.g., gestures, facial expressions), tactile inputs, or even audio signals (e.g., sounds from objects or voice) to sound, the project empowers users to create entirely new instruments, each with its own creative constraints ripe for exploration.

The complexity of these mappings introduces a learning curve, but also vast creative potential. A key advantage of the project’s flexible mapping approach is its adaptability: complex mappings can be simplified for beginners to quickly express themselves, with the option to gradually increase complexity as their skills develop. This scalability bridges the spectrum of musical expression, from effortless AI-generated music—where anyone can create high-quality songs with minimal input—to the nuanced, skill-intensive artistry of acoustic instruments. While AI-driven tools risk flooding the music space with low-effort content, the theremin Project seeks to explore the middle ground, fostering tools that balance accessibility with expressive depth.

Consider the extremes: on one end, AI allows someone with no musical training to generate a song by clicking a button or describing its style, offering high quality but limited personal expressivity. On the other, acoustic instruments demand years of practice to unlock their full potential, rewarding players with profound expressive control. The theremin Project envisions a platform where users can navigate this spectrum, creating instruments that adapt to their abilities while encouraging growth. By providing intuitive tools to map movement to sound, the project aims to democratize musical expression, uncover novel modes of creativity, and inspire a new generation of digital instruments. The theremin Project is a bold step toward this future, inviting exploration of the vast possibilities at the intersection of movement, sound, and technology.

## Work in Progress

The objective of this project is to enable users to easily make mappings between sensors and sound, 
in such a way that is advantageous for whatever their goal is: 
playing music, learning how to play music, making some sound producing installation, sonifying some data stream, etc.

Note that here were's starting with video and keyboard as the main "sensors", but the plan is to generalize this to all possible streams of input control. 
One of the main targets being controlling via sound itself. 
That is, giving our sound producing system the ability to hear as well. 
This, we hope, will open up many interesting abilities when paired with AI music gen, both for education and for improvisation.


## A few links:

See Leon theremin [playing his own instrument](https://www.youtube.com/watch?v=w5qf9O6c20o), 
or also here, [playing "Deep Night" (1930)](https://www.youtube.com/watch?v=WhR2e9ab-Uw).
The first one I even heard: 
[Samuel Hoffman playing "over the rainbox"](https://www.youtube.com/watch?v=K6KbEnGnymk). 
Someone might complain if I don't mention Led's Zepplin's Jimmy Page... [playing around](https://www.youtube.com/watch?v=KPhXm-UPfEU) with a theremin -- but personally, though I appreciate the exploration, I don't consider it as being "playing the theremin". 

# Old "about"

The objective of this project is to make "instruments" that work like a theremin, 
that is, whereby the features of the sounds produced are controlled by the position of the left and right hand. 

That, but generalized. 
A theremin is an analog device that whose sound is parametrized by two features only:
Pitch and volume, respectively controlled by the left and right hand; the 
distance from the device's sensor determines the pitch and volume of the sound produced.

Here, though, we'll use video as our sensor (and will consider more kinds of sensors later): So we can not only detect the positions of the hands, but also the positions of the fingers, and the angles between them, 
detect gestures, and so on. Further, we can use the facial expressions to add to 
the parametrization of the sound produced.

What the project wants to grow up to be is a platform for creating musical instruments
that are controlled by the body, enabling the user to determine the mapping between
the video stream and the sound produced. 
We hope this will enable the creation of new musical instruments that are more
intuitive and expressive than the traditional theremin.

Right now, the project is very much in its infancy, and we're just trying to get the basics working.

Here's a bit about what exists so far:

* `_03_hand_gesture_recognition_with_video_feedback.py`: Detects hand gestures and overlays
    the detected landmarks and categories on the video stream in real-time.
* `_06_hand_gesture_w_slabs.py`: The same, but using slabs, for a better framework setup
* `_05_a_wrist_lines.py`: Draws lines showing the wrist position on the video stream
    in real-time.
* `_07_wrist_lines_w_sound.py`: Draws lines showing the wrist position on the video stream
    and produces sound based on the wrist position. The resulting sound is horrible though, 
    due to the lack of a proper real-time setup (I believe). The sound produced is 
    produced in realtime, but only a small waveform chunk is produced. 
    It's choppy.
    **We need to buffer the stream of sound features and have an independent process that
    reads from the buffer and produces the sound, taking more than one chunk of 
    information to determine the sound produced.**
* `_08_keyboard_theremin.py`: To reduce the complexity of the sound produced, this script
    uses a keyboard to produce sound, with the pitch and volume controlled by the 
    left and right hand, respectively. This doesn't work well at all. More work is needed.




