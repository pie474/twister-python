import numpy as np
import matplotlib.pyplot as plt
import serial
import time
import random

# --- Config ---
USE_MOCK = False             # Set to False to use real serial data
SERIAL_PORT = '/dev/cu.usbserial-0001'
BAUD_RATE = 115200
ROWS, COLS = 18, 12
SHAPE = ROWS, COLS
MESSAGE_LENGTH = ROWS * COLS

BEGIN_SERIAL_SEQ = 'START'

def generate_mock_data():
    """Simulate a 12x18 matrix of sensor readings (0-1023)."""
    return np.random.randint(0, 1024, size=(ROWS, COLS))

debug = False
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

def visualize_heatmap(matrix):
    plt.clf()
    plt.imshow(matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=3000)
    plt.colorbar(label='Sensor Value')
    plt.title('Twister Sensor Heatmap')
    # plt.pause(0.1)

def main():
    if USE_MOCK:
        print("Running with mock data...")
        ser = None
    else:
        print("Connecting to serial...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Give time for Arduino to reset
        ser.read_until(BEGIN_SERIAL_SEQ)

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
