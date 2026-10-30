# theremin.framework.storage

Storage abstractions using dol for calibration and presets.

### Classes

| [`CalibrationStore`](#theremin.framework.storage.CalibrationStore)([base_path])   | Store for sensor calibration data using dict-like interface.   |
|----------------------------------------------------------------------------------|----------------------------------------------------------------|
| [`PresetStore`](#theremin.framework.storage.PresetStore)([base_path])        | Store for pipeline presets and configurations.                 |

### *class* theremin.framework.storage.CalibrationStore(base_path='./data/calibration')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Store for sensor calibration data using dict-like interface.

Uses dol for uniform storage abstraction, but falls back to simple
file-based storage if dol is not available.

### Example

```pycon
>>> store = CalibrationStore('./data/calibration')
>>> store['camera_0'] = {'offset_x': 0.1, 'offset_y': 0.2}
>>> calibration = store['camera_0']
```

#### keys()

Get all sensor names with calibration.

### *class* theremin.framework.storage.PresetStore(base_path='./data/presets')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Store for pipeline presets and configurations.

### Example

```pycon
>>> store = PresetStore('./data/presets')
>>> store['my_theremin'] = {
>>>     'pipeline': 'theremin',
>>>     'frequency_range': [200, 2000],
>>>     'amplitude_range': [0, 1]
>>> }
>>> preset = store['my_theremin']
```

#### keys()

Get all preset names.
