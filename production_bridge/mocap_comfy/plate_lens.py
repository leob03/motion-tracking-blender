"""Lens from an image-sequence header (focal length and filmback), header only; no pixels are read.

Production EXRs from cinema cameras carry the lens and sensor in their metadata, e.g. ARRI ALEXA 35:
FocalLength '32.77 mm', SensorPixelPitch '6.075000' (micrometres), SensorAcquisitionRectangle_Width '4608',
SensorDimensions '27.99 x 19.22 mm'. The focal length in pixels at the plate's width is
focal_mm / (pixel pitch x acquisition width) x plate width, which also holds when the EXR was scaled from the
acquisition size (not when it was cropped; that is reported when the sizes differ).
"""
import re
from pathlib import Path

FOCAL_KEYS = ('FocalLength', 'focalLength', 'lensFocalLength', 'camera/lens/focalLength', 'arri/lens/focalLength',
              'Exif:FocalLength', 'focal_length', 'Lens:FocalLength')
PITCH_KEYS = ('SensorPixelPitch', 'sensorPixelPitch')
ACQUISITION_KEYS = ('SensorAcquisitionRectangle_Width', 'StoredWidth', 'ActiveImageWidth')
SENSOR_KEYS = ('SensorDimensions', 'sensorDimensions', 'FilmbackWidth', 'filmbackWidth', 'camera/sensor/width', 'SensorWidth')
CAMERA_KEYS = ('CameraType', 'cameraModel', 'Make', 'Model', 'Exif:Model')


def _number(value):
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r'[-+]?\d*\.?\d+', str(value or ''))
    return float(match.group()) if match else None


def _first(attributes, keys):
    for key in keys:
        if key in attributes and attributes[key] not in (None, ''):
            return key, attributes[key]
    return None, None


def header_attributes(path):
    import OpenImageIO as oiio
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise ValueError(f'Cannot read {Path(path).name}')
    try:
        spec = image.spec()
        width = spec.full_width or spec.width
        attributes = {a.name: a.value for a in spec.extra_attribs}
    finally:
        image.close()
    return width, attributes


def read(path):
    """dict(focal_mm, sensor_width_mm, focal_px, plate_width, source, camera, warning) or None when the header has no lens."""
    try:
        width, attributes = header_attributes(path)
    except Exception:
        return None
    key, value = _first(attributes, FOCAL_KEYS)
    focal_mm = _number(value)
    if not focal_mm or focal_mm <= 0 or (attributes.get(key+'Present') == '0' and key != 'FocalLength'):
        return None
    warning = ''
    pitch_key, pitch = _first(attributes, PITCH_KEYS)
    acquisition_key, acquisition = _first(attributes, ACQUISITION_KEYS)
    pitch, acquisition = _number(pitch), _number(acquisition)
    if pitch and acquisition:
        sensor_mm = pitch*acquisition/1000.  # micrometres x pixels
        how = f'{key} {focal_mm:g} mm, {pitch_key} {pitch:g} µm x {int(acquisition)} px'
    else:
        sensor_key, sensor = _first(attributes, SENSOR_KEYS)
        sensor_mm = _number(sensor)
        if not sensor_mm:
            return dict(focal_mm=focal_mm, sensor_width_mm=None, focal_px=None, plate_width=width, camera=_camera(attributes),
                        source=f'{key} {focal_mm:g} mm (no filmback in the header)', warning='Filmback unknown: choose it below')
        how = f'{key} {focal_mm:g} mm, {sensor_key} {sensor}'
        acquisition = None
    stored = _number(attributes.get('StoredWidth'))
    if acquisition and stored and abs(stored-acquisition) > 1 and abs(stored/acquisition-width/max(stored, 1)) > .01:
        warning = f'Stored width {int(stored)} px differs from the acquisition width {int(acquisition)} px; check for a crop'
    return dict(focal_mm=focal_mm, sensor_width_mm=sensor_mm, focal_px=focal_mm/sensor_mm*width, plate_width=width,
                camera=_camera(attributes), source='EXR header: '+how, warning=warning)


def _camera(attributes):
    _, value = _first(attributes, CAMERA_KEYS)
    return str(value) if value else ''
