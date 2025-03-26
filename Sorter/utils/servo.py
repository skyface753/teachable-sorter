from gpiozero import Servo
from time import sleep


# Correction factor for the servo
myCorrection = 0.45
MAXPW = (2.0 + myCorrection) / 1000  # Maximum pulse width
MINPW = (1.0 - myCorrection) / 1000  # Minimum pulse width


class MyServo:
    def __init__(self, pin):
        self.pin = pin
        self.servo = Servo(pin, min_pulse_width=MINPW, max_pulse_width=MAXPW)

    def mid(self):
        self.servo.mid()

    def min(self):
        self.servo.min()

    def max(self):
        self.servo.max()


if __name__ == '__main__':
    # GPIO für Steuersignal
    servo_pin = 18
    # Servo-Objekt erstellen
    servo = MyServo(servo_pin)

    print('Position: Mitte (90 Grad)')
    servo.mid()
    sleep(2)

    print('Position: Ganz Links (0 Grad)')
    servo.min()
    sleep(2)

    print('Position: Mitte (90 Grad)')
    servo.mid()
    sleep(2)

    print('Position: Ganz Rechts (180 Grad)')
    servo.max()
    sleep(2)

    print('Position: Mitte (90 Grad)')
    servo.mid()
    sleep(2)

    print('Ende')
