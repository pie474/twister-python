import numpy as np
import matplotlib.pyplot as plt
import asyncio
import websockets
import json
import serial_asyncio
from time import sleep

from estimate_resistors import estimate_resistors, estimate_resistors_fast

# --- Config ---
USE_MOCK = False             # Set to False to use real serial data
ANTI_CROSSTALK = True
SERIAL_PORT = '/dev/cu.SLAB_USBtoUART'
BAUD_RATE = 115200

ROWS, COLS = 18, 12
SHAPE = ROWS, COLS
MESSAGE_LENGTH = ROWS * COLS

SERIAL_BEGIN_SEQ = 'START'

debug = True

def generate_mock_data():
    """Simulate a 12x18 matrix of sensor readings (0-1023)."""
    sleep(1)
    return np.random.randint(0, 3000, size=(ROWS, COLS), dtype='int32')

async def flush_stale_lines(reader: asyncio.StreamReader, timeout=0.01):
    """Flush buffered complete lines. Avoids cutting into partial lines."""
    flushed_count = 0
    try:
        while True:
            # Only flush if a complete line is present quickly
            line = await asyncio.wait_for(reader.readline(), timeout)
            flushed_count += 1
    except asyncio.TimeoutError:
        if flushed_count > 0:
            print(f"Flushed {flushed_count} stale lines")

async def read_clean_line(reader: asyncio.StreamReader) -> str | None:
    await flush_stale_lines(reader)

    try:
        raw_line = await reader.readline()
        return raw_line.decode('utf-8').strip()
    except UnicodeDecodeError:
        print("Decode error")
        return None

async def read_from_serial(ser):
    """
    Reads a 18x12 matrix of integers from serial.
    Expected format: 18*12 comma-separated values on a line.
    """
    line = await read_clean_line(ser)

    matrix = np.zeros(MESSAGE_LENGTH)

    if line:
        try:
            line = list(map(int, line.split(',')))
            if len(line) == MESSAGE_LENGTH:
                matrix = np.array(line, dtype='float')
        except ValueError:
            print('value error', line)
            pass
    return np.reshape(matrix, SHAPE)

async def main_loop(websocket):
    if USE_MOCK:
        print("Running with mock data...")
        ser = None
    else:
        print("Connecting to serial...")
        reader, writer = await serial_asyncio.open_serial_connection(
            url=SERIAL_PORT, baudrate=BAUD_RATE
        )
        ser = reader
        await ser.readuntil(SERIAL_BEGIN_SEQ.encode())
        print('Start sequence received')

    # plt.ion()
    # fig = plt.figure()

    try:
        while True:
            if USE_MOCK:
                matrix = generate_mock_data()

            else:
                matrix = (await read_from_serial(ser))

                if ANTI_CROSSTALK and not (matrix == 0).any():
                    estimated_resistances = estimate_resistors(R_eq=matrix, init='uniform', tol=1e-6, verbose=False)
                    # estimated_resistances = estimate_resistors_fast(R_eq=matrix, init='uniform', tol=1e-9, verbose=True, sparse=False)
                    if not estimated_resistances is None:
                        matrix = estimated_resistances

            metadata = {
                "shape": matrix.shape,
                "dtype": str(matrix.dtype)
            }
            await websocket.send(json.dumps(metadata))
            await websocket.send(matrix.tobytes())

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # plt.ioff()
        # plt.close()
        pass

async def main():
    async with websockets.serve(main_loop, "localhost", 8765):
        print("WebSocket server running at ws://localhost:8765")
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())
