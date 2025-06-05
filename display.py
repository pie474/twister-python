import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cairosvg
from io import BytesIO
from PIL import Image

SENSOR_REGION_IDS = {(0, 0): 'a', (1, 0): 'b', (2, 0): 'c', (0, 1): 'd', (1, 1): 'e', (2, 1): 'f', (0, 2): 'g', (1, 2): 'h', (2, 2): 'i', }

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