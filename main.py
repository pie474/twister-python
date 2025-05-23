import numpy as np
import matplotlib.pyplot as plt
import serial
import time
import random

# --- Config ---
USE_MOCK = True             # Set to False to use real serial data
SERIAL_PORT = 'COM3'        # Update this to match your port (e.g., '/dev/ttyUSB0')
BAUD_RATE = 9600
ROWS, COLS = 12, 18

def generate_mock_data():
    """Simulate a 12x18 matrix of sensor readings (0-1023)."""
    return np.random.randint(0, 1024, size=(ROWS, COLS))

def read_from_serial(ser):
    """
    Reads a 12x18 matrix of integers from serial.
    Expected format: 12 lines of comma-separated values, each with 18 integers.
    """
    matrix = []
    while len(matrix) < ROWS:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                row = list(map(int, line.split(',')))
                if len(row) == COLS:
                    matrix.append(row)
            except ValueError:
                continue
    return np.array(matrix)

def visualize_heatmap(matrix):
    plt.clf()
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Sensor Value')
    plt.title('Twister Sensor Heatmap')
    plt.pause(0.1)

def main():
    if USE_MOCK:
        print("Running with mock data...")
        ser = None
    else:
        print("Connecting to serial...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Give time for Arduino to reset

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
        plt.show()

if __name__ == "__main__":
    main()
