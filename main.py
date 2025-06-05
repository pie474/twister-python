import typing

import numpy as np
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
import matplotlib.colors as colors
import cairosvg
from io import BytesIO
from PIL import Image
import asyncio
import websockets
import json
import serial_asyncio

from estimate_resistors import estimate_resistors, estimate_resistors_fast

# --- Config ---
USE_MOCK = False             # Set to False to use real serial data
SERIAL_PORT = '/dev/cu.SLAB_USBtoUART'
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

async def read_from_serial(ser):
    """
    Reads a 18x12 matrix of integers from serial.
    Expected format: 18 lines of comma-separated values, each with 12 integers.
    """
    try:
        line = (await ser.readline()).decode('utf-8').strip()
    except UnicodeDecodeError:
        print('decode error')
        line = None

    matrix = np.zeros(MESSAGE_LENGTH)

    if line:
        try:
            line = list(map(int, line.split(',')))
            if len(line) == MESSAGE_LENGTH:
                matrix = np.array(line, dtype='float')
                # print(matrix[0])
        except ValueError:
            print('value error', line)
            pass
    return np.reshape(matrix, SHAPE)

def visualize_heatmap_as_grid(matrix):
    plt.clf()
    plt.imshow(matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=50000)
    plt.colorbar(label='Sensor Value')
    plt.yticks(np.arange(18))
    plt.xticks(np.arange(12))
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

async def main_loop(websocket):
    if USE_MOCK:
        print("Running with mock data...")
        ser = None
    else:
        print("Connecting to serial...")
        conn : typing.Tuple[asyncio.StreamReader, typing.Any] = await serial_asyncio.open_serial_connection(
            url=SERIAL_PORT, baudrate=BAUD_RATE
        )
        ser = conn[0]
        await ser.readuntil(BEGIN_SERIAL_SEQ.encode())
        print('Start sequence received')

    plt.ion()
    fig = plt.figure()

    try:
        while True:
            matrix = generate_mock_data() if USE_MOCK else (await read_from_serial(ser))

            if not (matrix == 0).any():
                estimated_resistances = estimate_resistors(R_eq=matrix, init=matrix, tol=1e-9, verbose=False)
                # estimated_resistances = estimate_resistors_fast(R_eq=matrix, init='uniform', tol=1e-9, verbose=True, sparse=False)
                print('solved')
                visualize_heatmap_as_grid(estimated_resistances)

            metadata = {
                "shape": matrix.shape,
                "dtype": str(matrix.dtype)
            }
            await websocket.send(json.dumps(metadata))
            await websocket.send(matrix.tobytes())

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        plt.ioff()
        plt.close()

async def main():
    async with websockets.serve(main_loop, "localhost", 8765):
        print("WebSocket server running at ws://localhost:8765")
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())
