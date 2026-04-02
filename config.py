"""
Konfigurasi untuk Person Tracker
"""

MODEL_PATH = "yolo11n.pt"

TRACKING_TIMEOUT = 5

COLORS = {
    'box': (0, 255, 0),
    'text_bg': (0, 255, 0),
    'text': (0, 0, 0),
    'info': (255, 255, 255)
}

FONT = {
    'face': 0,
    'scale': 0.6,
    'thickness': 2
}

DISPLAY = {
    'window_name': "Person Tracker - Time Monitoring",
    'show_fps': True
}

VIDEO = {
    'default_source': 0,
    'save_output': False,
    'output_path': "output.mp4"
}
