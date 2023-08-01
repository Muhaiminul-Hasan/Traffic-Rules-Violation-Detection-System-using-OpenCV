import datetime

import cv2
import imutils

# Font settings
font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.5
thickness = 1

# Texts


# Shadow color (you can adjust this to create the desired shadow effect)
shadow_color = (50, 50, 50)  # Dark gray



# import winsound

class TrafficProcessor:

    def __init__(self):
        self.firstFrame = None
        self.light = "Green"
        self.cnt = 0
        self.dynamic = False
        self.min_area = 500
        self.duration = 200  # millisecond
        self.freq = 900  # Hz

        # cam specific properties
        self.zone1 = (100, 150)
        self.zone2 = (450, 145)
        self.thres = 30
        self.dynamic = False

    def cross_violation(self, frame):

        isCar = False
        cropped_cars = []  # list for taking all violated car's snap

        # resize the frame, convert it to grayscale, and blur it
        if frame is None:
            return None
        self.frame = imutils.resize(frame, width=500)
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (21, 21), 0)

        # Texts
        ShowViolation = ""
        status_text = "Signal Status: {}".format(self.light)
        
        # Text position
        text_position1 = (7, 20)
        text_position2 = (7, 50)

        # if the first frame is None, initialize it
        if self.firstFrame is None:
            self.firstFrame = self.gray
            pack = {'frame': self.frame, 'reference': self.firstFrame, 'list_of_cars': cropped_cars, 'cnt': self.cnt}
            return pack

        # compute the absolute difference between the current frame and
        # first frame
        self.frameDelta = cv2.absdiff(self.firstFrame, self.gray)
        self.thresh = cv2.threshold(self.frameDelta, self.thres, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.min_area:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            if (self.zone1[0] < (x + w / 2) < self.zone2[0] and (y + h / 2) < self.zone1[1] + 100 and (
                y + h / 2) > self.zone2[1] - 100):
                    isCar = True

            if self.light == "Red" and self.zone1[0] < (x + w / 2) < self.zone2[0] and self.zone1[1] > (y + h / 2) > \
                self.zone2[1]:
                # winsound.Beep(self.freq, self.duration)
                rcar = self.frame[y:y + h, x:x + w]
                rcar = cv2.resize(rcar, (0, 0), fx=4, fy=4)
                cropped_cars.append(rcar)
                cv2.imwrite('reported_car/car_' + str(self.cnt) + ".jpg", rcar)
                self.cnt += 1
                ShowViolation = "<Violation>"

            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        if self.dynamic or not isCar:
            self.firstFrame = self.gray
        
        # draw the text and timestamp on the frame
        if self.light == "Green":
            color = (0, 200, 0)
        else:
            color = (0, 0, 200)

        cv2.rectangle(self.frame, self.zone1, self.zone2, (255, 0, 0), 2)
        
        # Show signal status
        shadow_position = (text_position1[0] + 2, text_position1[1] + 2)  # Shift the shadow slightly down and to the right
        cv2.putText(self.frame, "Signal Status: {}".format(self.light), shadow_position, font, font_scale, shadow_color, thickness, cv2.LINE_AA)
        
        cv2.putText(self.frame, "Signal Status: {}".format(self.light), text_position1,
                    font, font_scale, color, thickness)
        
        # Show violation
        shadow_position = (text_position2[0] + 2, text_position2[1] + 2)  # Shift the shadow slightly down and to the right
        cv2.putText(self.frame, ShowViolation, shadow_position, font, font_scale, shadow_color, thickness, cv2.LINE_AA)
        
        cv2.putText(self.frame, ShowViolation, text_position2,
                    font, font_scale, (0, 0, 255), thickness)
        
        # Show Date
        timestamp_position = (7, self.frame.shape[0] - 7)
        timestamp_scale = 0.4
        timestamp_thickness = 1
        timestamp_color = (255, 255, 255)
        
        shadow_position = (timestamp_position[0] + 2, timestamp_position[1] + 2)  # Shift the shadow slightly down and to the right
        cv2.putText(self.frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            shadow_position, font, timestamp_scale, shadow_color, timestamp_thickness, cv2.LINE_AA)


        cv2.putText(self.frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    timestamp_position, font, timestamp_scale, timestamp_color, timestamp_thickness)

        pack = {'frame': self.frame, 'reference': self.firstFrame, 'list_of_cars': cropped_cars, 'cnt': self.cnt}

        return pack
