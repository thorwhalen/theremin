# theremin.video_features

Hand gesture recognition and feature extraction for theremin.

### Functions

| [`calculate_angle`](#theremin.video_features.calculate_angle)(p1, p2, p3)                      | Calculate the angle between three points (p2 is the vertex).            |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`calculate_euclidean_distance`](#theremin.video_features.calculate_euclidean_distance)(point1, point2)     | Calculate the Euclidean distance between two 3D points.                 |
| [`many_single_hand_features`](#theremin.video_features.many_single_hand_features)(hand_landmarks[, ...]) | Extracts multiple hand features from a MediaPipe hand_landmarks object. |
| [`many_video_features`](#theremin.video_features.many_video_features)(hand_detection[, ...])       | Calls many_single_hand_features for each hand in the detection.         |

### Classes

| [`HandGestureRecognizer`](#theremin.video_features.HandGestureRecognizer)([mode, max_hands, ...])   | A class to detect hand gestures using MediaPipe's GestureRecognizer.   |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|

### *class* theremin.video_features.HandGestureRecognizer(mode=False, , max_hands=2, detection_con=0.5, track_con=0.5, gesture_recognizer_path='/home/runner/work/theremin/theremin/theremin/data/gesture_recognizer.task')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A class to detect hand gestures using MediaPipe’s GestureRecognizer.

#### mode

Mode for the hand detection.

* **Type:**
  [*bool*](https://docs.python.org/3/builtins/functions.html#bool)

#### max_hands

Maximum number of hands to detect.

* **Type:**
  [*int*](https://docs.python.org/3/builtins/functions.html#int)

#### detection_con

Minimum detection confidence threshold.

* **Type:**
  [*float*](https://docs.python.org/3/builtins/functions.html#float)

#### track_con

Minimum tracking confidence threshold.

* **Type:**
  [*float*](https://docs.python.org/3/builtins/functions.html#float)

#### find_hands(img)

Detects hands in the provided image.

* **Parameters:**
  **img** – The input image.
* **Returns:**
  The hand detection results

### theremin.video_features.calculate_angle(p1, p2, p3)

Calculate the angle between three points (p2 is the vertex).

### theremin.video_features.calculate_euclidean_distance(point1, point2)

Calculate the Euclidean distance between two 3D points.

### theremin.video_features.many_single_hand_features(hand_landmarks, include=frozenset({'finger_directions', 'gesture', 'gesture_score', 'handedness', 'index_finger_extension_angle1', 'index_finger_extension_angle2', 'is_pinching', 'openness', 'palm_center', 'palm_normal', 'thumb_index_distance', 'wrist_position'}), exclude=())

Extracts multiple hand features from a MediaPipe hand_landmarks object.

* **Parameters:**
  * **hand_landmarks** – MediaPipe hand landmarks
  * **include** – Set of features to include
  * **exclude** – Set of features to exclude
* **Returns:**
  Dictionary of extracted features
* **Return type:**
  [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)

### theremin.video_features.many_video_features(hand_detection, include=frozenset({'finger_directions', 'gesture', 'gesture_score', 'handedness', 'index_finger_extension_angle1', 'index_finger_extension_angle2', 'is_pinching', 'openness', 'palm_center', 'palm_normal', 'thumb_index_distance', 'wrist_position'}), exclude=())

Calls many_single_hand_features for each hand in the detection.
Returns a dictionary corresponding to the many_single_hand_features output,
but with `l_` and `r_` prefixes for left and right hands.

* **Parameters:**
  * **hand_detection** – MediaPipe hand detection results
  * **include** – Set of features to include
  * **exclude** – Set of features to exclude
* **Returns:**
  Dictionary of extracted features with hand prefixes
* **Return type:**
  [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)
