import numpy as np
import matplotlib.pyplot as plt
import serial
import time
import random

import xml.etree.ElementTree as ET
import matplotlib.colors as colors
import cairosvg
from io import BytesIO
from PIL import Image

# --- Config ---
USE_MOCK = True             # Set to False to use real serial data
SERIAL_PORT = '/dev/cu.usbserial-0001'
BAUD_RATE = 115200
ROWS, COLS = 18, 12
SHAPE = ROWS, COLS
MESSAGE_LENGTH = ROWS * COLS

BEGIN_SERIAL_SEQ = 'START'

SENSOR_REGION_IDS = {(0, 0): 'a', (1, 0): 'b', (2, 0): 'c', (0, 1): 'd', (1, 1): 'e', (2, 1): 'f', (0, 2): 'g', (1, 2): 'h', (2, 2): 'i', }
debug = True

# Parse SVG
SVG_TREE = ET.parse('sensor_array.svg')
SVG_ROOT = SVG_TREE.getroot()
SVG_NS = {'svg': 'http://www.w3.org/2000/svg'}  # standard SVG namespace

def generate_mock_data():
    """Simulate a 12x18 matrix of sensor readings (0-1023)."""
    return np.random.randint(0, 1024, size=(ROWS, COLS))

def read_from_serial(ser):
    """
    Reads a 18x12 matrix of integers from serial.
    Expected format: 18 lines of comma-separated values, each with 12 integers.
    """
    line = ser.readline().decode('utf-8').strip()
    matrix = np.zeros(MESSAGE_LENGTH)
    if line:
        try:
            line = list(map(int, line.split(',')))
            if len(line) == MESSAGE_LENGTH:
                matrix = np.array(line, dtype='int')
                print(matrix[0])
        except ValueError:
            print('value error')
            pass
    return np.reshape(matrix, SHAPE)

def visualize_heatmap_as_grid(matrix):
    plt.clf()
    plt.imshow(matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=3000)
    plt.colorbar(label='Sensor Value')
    plt.title('Twister Sensor Heatmap')
    plt.pause(0.1)

def apply_colors_to_svg(svg_path, sensor_values, colormap='viridis'):
    # Parse SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}  # standard SVG namespace

    norm = colors.Normalize(vmin=np.min(sensor_values), vmax=np.max(sensor_values))
    cmap = plt.get_cmap(colormap)

    for j, row in enumerate(sensor_values):
        for i, val in enumerate(row):
            id = f'{SENSOR_REGION_IDS[(j%3, i%3)]}_{j//3}_{i//3}'
            hex_color = colors.to_hex(cmap(norm(val)))
            path = root.find(f".//svg:*[@id='{id}']", ns)
            if path is not None:
                style = path.get('style')
                new_style = style.replace('fill:none', f'fill:{hex_color}')
                path.set('style', new_style)

    # Write modified SVG to string
    svg_bytes = ET.tostring(root)
    global debug
    if not debug:
        print(svg_bytes)
        debug = True
    return svg_bytes

def visualize_heatmap(matrix):
    svg_bytes = apply_colors_to_svg('sensor_array.svg', matrix)
    png_data = cairosvg.svg2png(bytestring=svg_bytes, scale=0.1, dpi=20)
    image = Image.open(BytesIO(png_data))

    plt.clf()
    plt.imshow(image)
    plt.axis('off')
    plt.colorbar(label='Sensor Value')
    plt.title('Twister Sensor Heatmap')
    plt.pause(0.1)

def main():
    if USE_MOCK:
        print("Running with mock data...")
        ser = None
    else:
        print("Connecting to serial...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
        time.sleep(2)  # Give time for Arduino to reset
        ser.read_until(BEGIN_SERIAL_SEQ)
        print('Start sequence received')

    plt.ion()
    fig = plt.figure()

    try:
        while True:
            matrix = generate_mock_data() if USE_MOCK else read_from_serial(ser)
            visualize_heatmap(matrix)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if ser:
            ser.close()
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    main()
