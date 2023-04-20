# Use hand to control mycobot
import cv2
from HandTrackingModule import HandDetector
from pymycobot.mycobot import MyCobot
from pymycobot.genre import Angle
from pymycobot import PI_PORT, PI_BAUD # 当使用树莓派版本的mycobot时，可以引用这两个变量进行MyCobot初始化
import time


class Main:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.detector = HandDetector()
        # self.camera.set(3, 640)
        # self.camera.set(4, 480)

    def Gesture_recognition(self):
        last_avg_m = 0
        last_avg_n = 0
        cnt = 0
        while True:

            frame, img = self.camera.read()
            img = self.detector.findHands(img)
            lmList, bbox = self.detector.findPosition(img)

            if lmList:
                cnt = cnt+1
                x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
                x1, x2, x3, x4, x5 = self.detector.fingersUp()
                if x2 == 1 and (x1 == 0 and x3 == 0 and x4 == 0 and x5 == 0) and cnt > 5:
                    cv2.putText(img, "1_ONE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)
                    mc.send_angles(
                            [-1.49, 115, -153.45, 30, -33.42, 137.9], 50)
                    mc.set_color(0, 0, 50)
                    time.sleep(1.5)
                    for i in range(5):
                        mc.send_angles(
                            [-1.49, 115, -153.45, 30, -33.42, 137.9], 80)
                        mc.set_color(0, 0, 50)
                        time.sleep(0.8)
                        mc.send_angles(
                            [-1.49, 55, -153.45, 80, 33.42, 137.9], 80)
                        mc.set_color(0, 50, 0)
                        time.sleep(0.8)
                    mc.send_angles([90, 0, 0, 0, -90, -45], 80)
                    time.sleep(1)
                    cnt = 0
                elif (x2 == 1 and x3 == 1) and (x4 == 0 and x5 == 0 and x1 == 0):
                    cv2.putText(img, "2_TWO", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)

                elif (x2 == 1 and x3 == 1 and x4 == 1) and (x1 == 0 and x5 == 0):
                    cv2.putText(img, "3_THREE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)

                elif (x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1) and (x1 == 0):
                    cv2.putText(img, "4_FOUR", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)

                elif x1 == 1 and x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1:
                    m1, m2, m3, m4, m5 = lmList[0][0], lmList[5][0], lmList[9][0], lmList[13][0], lmList[17][0]
                    n1, n2, n3, n4, n5 = lmList[0][1], lmList[5][1], lmList[9][1], lmList[13][1], lmList[17][1]
                    avg_m = (m1 + m2 + m3 + m4 + m5) / 5
                    avg_n = (n1 + n2 + n3 + n4 + n5) / 5
                    if last_avg_m == 0 and last_avg_n == 0:
                        last_avg_m, last_avg_n = avg_m, avg_n

                    cv2.circle(img, (int(avg_m), int(avg_n)),
                               15, (255, 0, 0), cv2.FILLED)
                    if abs(avg_m - last_avg_m) > 70 or abs(avg_n - last_avg_n) > 70:
                        # 上下左右,no up and down currently
                        if avg_m - last_avg_m > 70 and abs(avg_n - last_avg_n) < 70 and cnt > 5:
                            # print("左")
                            cv2.putText(img, "Left", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                        (0, 0, 255), 3)
                            # mc.send_angles([90, 50, 0, -60, -90, -45], 50)
                            mc.send_angles([90, 50, 0, -60, -60, -45], 50)
                            cnt = 0
                            # time.sleep(1)
                        elif last_avg_m - avg_m > 70 and abs(avg_n - last_avg_n) < 70 and cnt > 5:

                            cv2.putText(img, "right", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                        (0, 0, 255), 3)
                            # mc.send_angles([90, -50, 0, 60, -90, -50], 50)
                            mc.send_angles([90, -50, 0, 60, -120, -50], 50)
                            cnt = 0
                            # time.sleep(1)
                        last_avg_m, last_avg_n = avg_m, avg_n
                elif x1 == 1 and x2 == 1 and x3 == 0 and x4 == 0 and x5 == 1:  # spiderman-stop啦
                    break

            # cv2.imshow("camera", cv2.resize(img,   None, fx=0.33,
            #            fy=0.33, interpolation=cv2.INTER_AREA))
            cv2.imshow("camera", img)
            if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
                break
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == '__main__':
    mc = MyCobot(PI_PORT, PI_BAUD)
    mc.send_angles([0, 0, 0, 0, -90, -45], 50)
    time.sleep(2.5)
    mc.send_angle(Angle.J1.value, 90, 50)
    time.sleep(2.5)
    Solution = Main()
    Solution.Gesture_recognition()
    Solution.camera.release()
    time.sleep(2.5)
    # mc.release_all_servos()
