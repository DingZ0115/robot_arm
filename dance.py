from pymycobot.mycobot import MyCobot
from pymycobot import PI_PORT, PI_BAUD
import time

if __name__ == '__main__':
    mc = MyCobot(PI_PORT, PI_BAUD)
    mc.send_angles([90, -50, 0, 60, -120, -50], 50)
    time.sleep(2)
    # start = time.time()
    # mc.send_angles([-1.49, 115, -153.45, 30, -33.42, 137.9], 80)
    # while not mc.is_in_position([-1.49, 115, -153.45, 30, -33.42, 137.9], 0):
    #     mc.resume()
    #     time.sleep(0.5)
    #     mc.pause()
    #     if time.time()-start > 3:
    #         break

    # start = time.time()
    # while time.time()-start < 3:
    #     mc.send_angles([-1.49, 115, -153.45, 30, -33.42, 137.9], 80)
    #     mc.set_color(0, 0, 50)
    #     time.sleep(1)
    #     mc.send_angles([-1.49, 55, -153.45, 80, 33.42, 137.9], 80)
    #     mc.set_color(0, 50, 0)
    #     time.sleep(1)
    # mc.release_all_servos()
