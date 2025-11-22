"""Storage abstractions using dol for calibration and presets."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import time


class CalibrationStore:
    """
    Store for sensor calibration data using dict-like interface.

    Uses dol for uniform storage abstraction, but falls back to simple
    file-based storage if dol is not available.

    Example:
        >>> store = CalibrationStore('./data/calibration')
        >>> store['camera_0'] = {'offset_x': 0.1, 'offset_y': 0.2}
        >>> calibration = store['camera_0']
    """

    def __init__(self, base_path: str = './data/calibration'):
        """
        Initialize calibration store.

        Args:
            base_path: Base directory for calibration files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._dol_available = False

        # Try to import dol
        try:
            from dol import Store
            self._Store = Store
            self._dol_available = True
        except ImportError:
            pass

    def _id_of_key(self, sensor_name: str) -> Path:
        """Convert sensor name to file path."""
        return self.base_path / f"calibration_{sensor_name}.json"

    def __getitem__(self, sensor_name: str) -> Dict:
        """Get calibration data for sensor."""
        file_path = self._id_of_key(sensor_name)

        if not file_path.exists():
            raise KeyError(f"No calibration data for {sensor_name}")

        with open(file_path, 'r') as f:
            return json.load(f)

    def __setitem__(self, sensor_name: str, calibration_data: Dict):
        """Set calibration data for sensor."""
        file_path = self._id_of_key(sensor_name)

        with open(file_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

    def __contains__(self, sensor_name: str) -> bool:
        """Check if calibration exists for sensor."""
        return self._id_of_key(sensor_name).exists()

    def keys(self):
        """Get all sensor names with calibration."""
        return [
            f.stem.replace('calibration_', '')
            for f in self.base_path.glob('calibration_*.json')
        ]


class PresetStore:
    """
    Store for pipeline presets and configurations.

    Example:
        >>> store = PresetStore('./data/presets')
        >>> store['my_theremin'] = {
        >>>     'pipeline': 'theremin',
        >>>     'frequency_range': [200, 2000],
        >>>     'amplitude_range': [0, 1]
        >>> }
        >>> preset = store['my_theremin']
    """

    def __init__(self, base_path: str = './data/presets'):
        """
        Initialize preset store.

        Args:
            base_path: Base directory for preset files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _id_of_key(self, preset_name: str) -> Path:
        """Convert preset name to file path."""
        return self.base_path / f"{preset_name}.json"

    def __getitem__(self, preset_name: str) -> Dict:
        """Get preset configuration."""
        file_path = self._id_of_key(preset_name)

        if not file_path.exists():
            raise KeyError(f"No preset named {preset_name}")

        with open(file_path, 'r') as f:
            return json.load(f)

    def __setitem__(self, preset_name: str, preset_data: Dict):
        """Set preset configuration."""
        # Add metadata
        preset_data['_created'] = time.time()

        file_path = self._id_of_key(preset_name)

        with open(file_path, 'w') as f:
            json.dump(preset_data, f, indent=2)

    def __contains__(self, preset_name: str) -> bool:
        """Check if preset exists."""
        return self._id_of_key(preset_name).exists()

    def keys(self):
        """Get all preset names."""
        return [f.stem for f in self.base_path.glob('*.json')]
