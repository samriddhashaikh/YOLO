from gpiozero import LED
from time import sleep
from machine import I2C, Pin
from lcd1602 import LCD

# Initialize LED pins for Lane 1 and Lane 2
lane1_red = LED(27)     # GPIO 27 (BOARD 13)
lane1_yellow = LED(24)  # GPIO 24 (BOARD 18)
lane1_green = LED(17)   # GPIO 17 (BOARD 11)

lane2_red = LED(23)     # GPIO 23 (BOARD 16)
lane2_yellow = LED(25)  # GPIO 25 (BOARD 22)
lane2_green = LED(22)   # GPIO 22 (BOARD 15)

# Initialize I2C LCD display
# I2C-1 bus on Raspberry Pi: SDA = GPIO2 (Pin 3), SCL = GPIO3 (Pin 5)
i2c = I2C(1, scl=Pin(3), sda=Pin(2))
lcd = LCD(i2c, addr=0x27)

def traffic_cycle(red_led, yellow_led, green_led):
    # RED for 7s; yellow overlaps last 3s
    red_led.on()
    sleep(4)
    yellow_led.on()
    sleep(3)
    red_led.off()
    yellow_led.off()

    # GREEN for 7s
    green_led.on()
    sleep(7)
    green_led.off()

def clear_all_leds():
    for led in [lane1_red, lane1_yellow, lane1_green, lane2_red, lane2_yellow, lane2_green]:
        led.off()

def display_message():
    lcd.clear()
    lcd.write(0, 0, "Hello Soharab")
    lcd.write(1, 0, "Hello Sami")
    sleep(5)

def main():
    try:
        clear_all_leds()

        print("Running Lane 1 sequence...")
        traffic_cycle(lane1_red, lane1_yellow, lane1_green)

        sleep(2)

        print("Running Lane 2 sequence...")
        traffic_cycle(lane2_red, lane2_yellow, lane2_green)

        # Display message on LCD
        display_message()

    finally:
        clear_all_leds()
        lcd.clear()
        lcd.write(0, 0, "Test Complete")
        sleep(2)
        lcd.clear()

if __name__ == "__main__":
    main()
