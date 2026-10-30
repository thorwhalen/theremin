# theremin.display

Display utilities for theremin visualization.

### Functions

| [`display_sound_features_on_image`](#theremin.display.display_sound_features_on_image)(img, ...[, ...])   | Display sound features on the image with a semi-transparent background.                |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`draw_hand_landmarks`](#theremin.display.draw_hand_landmarks)(img, hand_detection, ...)      | Draw hand landmarks on the image.                                                      |
| [`draw_on_screen`](#theremin.display.draw_on_screen)(recognizer, img, hand_detection)    | Draw hand landmarks, wrist lines, and sound features on the image.                     |
| [`draw_wrist_lines`](#theremin.display.draw_wrist_lines)(img, hand_detection, mp_hands)    | Draws vertical and horizontal lines from the edges of the image to the wrist position. |

### theremin.display.display_sound_features_on_image(img, sound_features, , font=0, font_scale=1.0, color=(0, 255, 0), thickness=2, float_format='.2f', x_pos=10, y_pos=30, y_increment=30, bg_color=(150, 150, 150, 128))

Display sound features on the image with a semi-transparent background.

* **Parameters:**
  * **img** (`ndarray`) – The image to draw on
  * **sound_features** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary of sound features
  * **font** – Font type to use
  * **font_scale** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Size of the font
  * **color** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]]) – Text color in BGR format
  * **thickness** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Line thickness of text
  * **float_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Format string for float values
  * **x_pos** – Starting x position for text
  * **y_pos** – Starting y position for text
  * **y_increment** – Vertical space between lines
  * **bg_color** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]]) – Background color (BGR + alpha) where alpha is 0-255

### theremin.display.draw_hand_landmarks(img, hand_detection, mp_hands, mp_draw)

Draw hand landmarks on the image.

* **Parameters:**
  * **img** – The image to draw on
  * **hand_detection** – MediaPipe hand detection results
  * **mp_hands** – MediaPipe hands module
  * **mp_draw** – MediaPipe drawing utilities
* **Returns:**
  The image with landmarks drawn
* **Return type:**
  img

### theremin.display.draw_on_screen(recognizer, img, hand_detection, sound_features=None, \*, draw_landmarks=True, draw_sound_features=<function display_sound_features_on_image>, draw_frequencies=None)

Draw hand landmarks, wrist lines, and sound features on the image.

* **Parameters:**
  * **recognizer** – HandGestureRecognizer instance
  * **img** (`ndarray`) – The input image
  * **hand_detection** – Hand detection results
  * **sound_features** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Sound parameter values to display
  * **draw_landmarks** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to draw hand landmarks
  * **draw_sound_features** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Function to draw sound features (or None to skip)
  * **draw_frequencies** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Iterable of frequencies to display as scale points on screen
* **Returns:**
  The image with visualizations added
* **Return type:**
  img

### theremin.display.draw_wrist_lines(img, hand_detection, mp_hands)

Draws vertical and horizontal lines from the edges of the image to the wrist position.

* **Parameters:**
  * **img** – The input image.
  * **hand_detection** – The hand detection results.
  * **mp_hands** – MediaPipe hands module
* **Returns:**
  The image with lines drawn.
* **Return type:**
  img
