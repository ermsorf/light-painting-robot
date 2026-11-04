import serial

ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)

def send_color(r, g, b):
    ser.write(f"{r},{g},{b}\n".encode())

send_color(255, 0, 127) 
