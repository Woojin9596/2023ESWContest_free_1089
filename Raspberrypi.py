import RPi.GPIO as GPIO
import time
import numpy as np
import serial
import cv2
import math
from gpiozero import Motor


ser = serial.Serial('/dev/serial0', 9600, timeout=0.5)

trigPin = 2   # GPIO 14
echoPin = 3   # GPIO 

cylinder_motor = Motor(forward=23, backward=24)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(trigPin, GPIO.OUT)
GPIO.setup(echoPin, GPIO.IN)

car_height = 18
car_speed_kmh = 0.5
pothole_depth = []
num_measurements = 5
measurements = np.full(num_measurements, 40.0)  # Initialize with 40cm to avoid division by zero

# 자율주행 차량 세팅
cap = cv2.VideoCapture(0)

# ROI (Region of Interest) 설정
roi_top = 300  # 상단 y 좌표
roi_bottom = 480  # 하단 y 좌표

# # 녹화를 위한 설정
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# out = cv2.VideoWriter('/home/cho/Videos/lane_detection.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

# 초음파 센서 signal
signal_received = False
total_pothole_time = 0
ultrasonic_time = 0 

# def get_distance_ultrasonic():
#     GPIO.output(trigPin, GPIO.LOW)
#     time.sleep(0.002)

#     GPIO.output(trigPin, GPIO.HIGH)
#     time.sleep(0.02)
#     GPIO.output(trigPin, GPIO.LOW)

#     while GPIO.input(echoPin) == GPIO.LOW:
#         pulse_start = time.time()

#     while GPIO.input(echoPin) == GPIO.HIGH:
#         pulse_end = time.time()

#     pulse_duration = pulse_end - pulse_start
#     distance = pulse_duration * 17150
#     distance = round(distance, 2)
    
#     return distance

def get_distance_ultrasonic():
    GPIO.output(trigPin, GPIO.LOW)
    time.sleep(0.002)

    GPIO.output(trigPin, GPIO.HIGH)
    start_pulse = time.time()
    time.sleep(0.02)
    GPIO.output(trigPin, GPIO.LOW)

    while GPIO.input(echoPin) == GPIO.LOW:
        pass
    start_echo = time.time()

    while GPIO.input(echoPin) == GPIO.HIGH:
        pass
    end_echo = time.time()

    pulse_duration = end_echo - start_echo
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    
    ultrasonic_time = start_echo - start_pulse

    return distance, ultrasonic_time


def moving_average(new_measurement):
    global measurements
    measurements[1:] = measurements[:-1]  # Shift measurements to the left
    measurements[0] = new_measurement
    avg_distance = np.mean(measurements)
    return avg_distance


## 모터 제어 ##

class MotorController:
    def __init__(self, rpwm_pin, l_en_pin, r_en_pin):
        self.rpwm_pin = rpwm_pin
        self.l_en_pin = l_en_pin
        self.r_en_pin = r_en_pin
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        GPIO.setup(self.rpwm_pin, GPIO.OUT)
        GPIO.setup(self.l_en_pin, GPIO.OUT)
        GPIO.setup(self.r_en_pin, GPIO.OUT)
        
        GPIO.output(self.r_en_pin, True)
        GPIO.output(self.l_en_pin, True)
        
        self.rpwm = GPIO.PWM(self.rpwm_pin, 100)
        
        self.rpwm.start(0)
    
    def set_speed(self, speed):
        self.rpwm.ChangeDutyCycle(speed)
    
    def cleanup(self):
        self.rpwm.stop()
        GPIO.cleanup()

motor_R = MotorController(12, 20, 21)
motor_L = MotorController(13, 5, 6)

def move_forward():
    motor_R.set_speed(15)
    motor_L.set_speed(18)
    print("go")

def move_left():
    motor_R.set_speed(45)
    motor_L.set_speed(5)
    print("left")
    
def move_right():
    motor_R.set_speed(5)
    motor_L.set_speed(45)
    print("right")

# def move_pour():   # 조정 필요
#     motor_R.set_speed(10)
#     motor_L.set_speed(15)

def motor_stop():
    motor_R.set_speed(0)
    motor_L.set_speed(0)

def pour_sand():  
    # 모래 붓기 시작
    print("The sand is being poured!")
    cylinder_motor.forward(speed=0.9)  # 모터 속도 조정
    time.sleep(3)  # 모래 붓는 시간 조정
    cylinder_motor.stop()  # 모래 붓기 중단
    time.sleep(3)  # 모래 붓는 시간 조정
    cylinder_motor.backward(speed=0.9)  # 모터 속도 조정
    time.sleep(3)  # 모래 붓는 시간 조정
    cylinder_motor.stop()  # 모터 정지
    print("The sand has been poured!")
    time.sleep(3)



def detect_road_condition(distance):
    if distance >= 22:
        return "Pothole"
    elif 17 <= distance <= 19:
        return "Road"
    else:
        return "Unknown"

def calculate_pothole_volume(car_speed_kmh, total_pothole_time, average_pothole_depth):
    # 변환 상수 및 계산
    car_speed_mps = car_speed_kmh * 1000 / 3600  # 자동차의 속도 (m/s)
    cylinder_volume_conversion = 1000000  # cm^3 to m^3 변환 상수
    
    # 포트홀의 부피 계산
    cylinder_radius = average_pothole_depth / 2  # 포트홀의 반지름 (깊이의 절반)
    cylinder_height = car_speed_mps * total_pothole_time  # 움직이는 동안 포트홀을 지난 높이
    cylinder_volume_cm3 = np.pi * np.square(cylinder_radius) * cylinder_height  # 부피 (cm^3)
    # cylinder_volume_m3 = cylinder_volume_cm3 / cylinder_volume_conversion  # 부피 (m^3)
    
    return cylinder_volume_cm3
    
try:
    while True:
        received_data = ser.readline().decode().strip()
        total_pothole_time = 0
        if received_data == "detected_pothole":
            signal_received = True
            while True: 
                move_forward()

                # 초음파 데이터값 읽어오기
                if signal_received == True:
                    raw_distance, ultrasonic_time = get_distance_ultrasonic()

                    # Limit raw_distance to the range of 15-35cm
                    corrected_distance = min(max(raw_distance, 15.0), 35.0)

                    # If corrected_distance is over 50cm, set it to 20cm
                    if corrected_distance > 50.0:
                        corrected_distance = 20.0

                    prev_distance = averaged_distance
                    averaged_distance = moving_average(corrected_distance)

                    condition = detect_road_condition(averaged_distance)

                    if condition == "Pothole":    # 포트홀 진입 (숫자 조정 필요)
                        # print("Condition is Pothole")
                        print("Found the pothole location")
                        time.sleep(1)
                        pothole_depth.append(averaged_distance - car_height)
                        total_pothole_time += ultrasonic_time
                        # print("Pothole depth: {0:.2f}cm" .format(pothole_depth))

                    elif condition == "Road" and len(pothole_depth):
                        # print("Condition is Road")
                        move_forward()
                        time.sleep(3)
                        motor_stop()
                        average_pothole_depth = round(sum(pothole_depth) / len(pothole_depth), 2)
                        pothole_volume = round(calculate_pothole_volume(car_speed_kmh, total_pothole_time, average_pothole_depth), 2)
                        print("Pothole's Volume: ", pothole_volume, "cm^3")
                        print("The average depth of potholes is {0:.2f}cm" .format(average_pothole_depth))
                        pour_sand()
                        pothole_depth.clear()
                        break

        ret, frame = cap.read()
        if not ret:
            break
    
        # ROI 설정
        roi_frame = frame[roi_top:roi_bottom, :]
    
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    
        # Canny 엣지 검출 추가
        # threshold2 값이 클수록 확실한 edge   
        edges = cv2.Canny(gray, threshold1=400, threshold2=500)
        
        # 노이즈 제거를 위한 블러링
        blurred = cv2.GaussianBlur(edges, (5, 5), 0)
        
        # White mask 제작
        lower_white = np.array([220, 220, 220])  
        upper_white = np.array([255, 255, 255])  
        white_mask = cv2.inRange(frame, lower_white, upper_white)
        
        # minLength: 50, maxlinegap: 100
        lines = cv2.HoughLinesP(white_mask, rho=1, theta=np.pi/180, threshold=50, minLineLength=150, maxLineGap=30)
        
        if lines is not None:  # 검출된 선이 있는 경우에만 처리
            line_image = np.zeros_like(frame)
            
            # 흰색 차선들의 교점 계산
            intersections = []
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    x1, y1, x2, y2 = lines[i][0]
                    x3, y3, x4, y4 = lines[j][0]
                    
                    # 두 직선의 교점 계산
                    if (x2 - x1) != 0 and (x4 - x3) != 0:  # 직선이 수직이 아닌 경우에만 계산
                        intersection_x = np.int64((y3 - y1 + (x1 * y2 - x2 * y1) / (x2 - x1)) / ((y2 - y1) / (x2 - x1)))
                        intersection_y = np.int64((y2 - y1) / (x2 - x1) * (intersection_x - x1) + y1)
                        intersections.append((intersection_x, intersection_y))
                        
                    else:
                        continue   # 직선이 수직이면 계산하지 않고 넘어감
            
            if intersections:  # 교점이 존재하는 경우
                # 교점들의 평균 계산
                avg_intersection = np.mean(intersections, axis=0, dtype=np.int64)
                if intersections and not np.isnan(avg_intersection).any() and (0 <= avg_intersection[0] < frame.shape[1]) and (0 <= avg_intersection[1] < frame.shape[0]):
                    cv2.circle(line_image, (int(avg_intersection[0]), int(avg_intersection[1])), 10, (255, 0, 0), -1)
        # 이후의 코드와 마찬가지로 나머지 처리를 계속 진행


            
                # 중앙 교점 x 좌표를 기준으로 판단
                if avg_intersection[0] < frame.shape[1] // 2:
                    direction = "Left Turn"
                    move_left()
                # elif avg_intersection[0] > frame.shape[1] // 2:
                #     direction = "Right Turn"
                #     move_right()
                else:
                    direction = "Straight"
                    move_forward()
               
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(line_image, direction, (30, 30), font, 1, (0, 255, 0), 2)
            
            # 차선 그리기 (빨간색)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                
            result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        else:
            result = frame
        
        cv2.imshow("Lane Detection", result)
        # out.write(result)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


except KeyboardInterrupt:
    GPIO.cleanup()
    cap.release()
    # out.release()
    cv2.destroyAllWindows()



