import torch
import numpy as np
from PIL import Image
import time
import threading

from jetcam.csi_camera import CSICamera
from cnn.center_dataset import TEST_TRANSFORMS
from torchvision.models import alexnet
from base_ctrl import BaseController
from ultralytics import YOLO
from collections import deque
# ========== 제어 파라미터 ==========
MAX_STEER = 4.0
MAX_SPEED = 0.5
Kp = 0.015  # 비례 제어 게인
MOVING_AVG_WINDOW = 3
error_buffer = deque(maxlen=MOVING_AVG_WINDOW)
steering_buffer = deque(maxlen=3)


# ========== 장치 초기화 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model():
    model = alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 2)
    return model

model = get_model()

model.load_state_dict(torch.load('/home/ircv15/em7/perception/road_following_model43_best.pth'))
model.to(device)
model.eval()

yolo_model = YOLO('/home/ircv15/em7/perception/traffic_light_sign0521.pt')
yolo_model_task3 = YOLO('/home/ircv15/em7/perception/cars_detect.pt')
yolo_model_task4 = YOLO('/home/ircv15/em7/perception/task4_model02.pt')
class_names = {int(k): v.capitalize() for k, v in yolo_model.names.items()}
class_names_task3 = {int(k): v.capitalize() for k, v in yolo_model_task3.names.items()}
class_names_task4 = {int(k): v for k, v in yolo_model_task4.names.items()}
min_w, min_h = 0.03, 0.11

# ========== 상태 변수 ==========
switching_count = 0
green_start_time = None
# prev_sig = "Green"
# cur_sig = "Green"
task_one_stop_and_go = False
task_one_clear = True
task_two_clear = True
task_three_clear = False
task_four_clear = False
sig_task4_clear = False
task4_sig_start = False
#task4
is_left = False
is_right = False
is_straight = False
task_four_stop_and_go = False
# Task 2 action timer
action_label = None
action_start_time = 0.0
detected = False
cur_sig = "None"

# ========== 유틸리티 함수 ==========
def preprocess(image: np.ndarray):
    image_pil = Image.fromarray(image)
    tensor = TEST_TRANSFORMS(image_pil).to(device)
    return tensor.unsqueeze(0)

def clip(val, max_val):
    return max(min(val, max_val), -max_val)

# ========== 제어 함수 ==========
base = BaseController('/dev/ttyUSB0', 115200)

def run_motion(steering, speed, duration_sec):
    start = time.time()
    while time.time() - start < duration_sec:
        update_vehicle_motion(steering, speed)
        time.sleep(0.033)

def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
    threading.Thread(target=worker).start()

def update_vehicle_motion(steering, speed):
    steer_val = -clip(steering, MAX_STEER)
    speed_val = clip(speed, MAX_SPEED)

    base_speed = abs(speed_val)
    left_ratio = 1.0 * max(1.0 - steer_val, 0)
    right_ratio = 1.0 * max(1.0 + steer_val, 0)

    L = clip(base_speed * left_ratio, MAX_SPEED)
    R = clip(base_speed * right_ratio, MAX_SPEED)

    if speed_val > 0:
        L, R = -L, -R

    send_control_async(-L, -R)
    # print(f"[UGV] Steering={steer_val:.2f}, Speed={speed_val:.2f} → L={L:.2f}, R={R:.2f}", flush=True)

# ========== 카메라 연결 ==========
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)
time.sleep(2.0)

for _ in range(10):
    _ = camera.read()
    time.sleep(0.05)

# ========== 메인 루프 ==========
try:
    # if not task_one_clear:
    #     fixed_speed = 0.12
    # else:
    fixed_speed = 0.18    

    while True:
        frame = camera.read()
        height, width = frame.shape[:2]
        cur_time = time.time()

        # =================== 중앙선 예측 ===================
        with torch.no_grad():
            input_tensor = preprocess(frame)
            output = model(input_tensor).cpu().numpy()[0]

        x = (output[0] / 2 + 0.5) * width
        y = (output[1] / 2 + 0.5) * height

        if x > 570:
            x_center = 570
        elif x < 470:
            x_center = 470
        else:
            x_center = 520

        x_error = x_center - 520
        error_buffer.append(x_error)
        Ki=0.018
        # 이동 평균 적용
        avg_error = np.mean(error_buffer)
        steering = clip(Kp * avg_error+Ki*avg_error*0.0333, MAX_STEER)
        # steering = clip(Kp * x_error, MAX_STEER)

        # =================== 객체 인식 ===================
        yolo_result = yolo_model.predict(frame, conf=0.5, verbose=False)[0]
        yolo_result_task3 = yolo_model_task3.predict(frame, conf=0.5, verbose=False)[0]
        yolo_result_task4 = yolo_model_task4.predict(frame, conf=0.4, verbose=False)[0]
        # cur_sig = "None"
        speed = fixed_speed

        # Task 1: 신호등 처리
        if not task_one_clear:
            steering*=0.2
            for box in yolo_result.boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                cx, cy, bw, bh = box.xywhn[0].tolist()

                if label in ['Red', 'Green'] and bh > min_h*0.8:
                    detected = True
                    cur_sig = label
                    break

            # Red 신호 감지 시 정지 및 stop_and_go 플래그 설정
            if detected and cur_sig == "Red":
                speed = 0.0
                steering *= 0.3
                task_one_stop_and_go = True
                green_start_time = None  # Green 타이머 초기화

            # Green 신호가 감지되었고, Red를 거쳐온 경우 → 통과
            elif detected and task_one_stop_and_go and cur_sig == "Green":
                task_one_clear = True
                detected = False
                cur_sig = "None"
                steering*=0.3
                green_start_time = None

            # Green만 계속 감지되는 경우 (Red 안 나옴)
            elif detected and cur_sig != "Red":
                if green_start_time is None:
                    green_start_time = time.time()
                elif time.time() - green_start_time > 4.5:
                    print('green! skip!')
                    task_one_clear = True
                    detected = False
                    cur_sig = "None"
                    steering*=0.3
                    green_start_time = None
                                


        # Task 2: 정지표지판, 느린표지판
        elif not task_two_clear:
            if action_label is not None:
                if cur_time - action_start_time < 1.5:
                    cur_sig = action_label
                    detected = True
                    if action_label == "Stop":
                        speed = 0.0
                    elif action_label == "Slow":
                        speed = 0.12
                        steering*=0.5
                else:
                    task_two_clear = True
                    detected= False
                    action_label = None
            # if not detected and not task_two_clear:
            if not detected:
                for box in yolo_result.boxes:
                    cls_id = int(box.cls[0])
                    label = class_names[cls_id]
                    cx, cy, bw, bh = box.xywhn[0].tolist()

                    if label in ['Stop', 'Slow'] and bh > 0.1:
                        action_label = label
                        action_start_time = cur_time
                        cur_sig = label
                        detected = True
                        if label == "Stop":
                            speed = 0.0
                        elif label == "Slow":
                            speed = 0.1
                        break

        elif not task_three_clear:
            print('this3')
            turn_left = False
            turn_right= False
            # if not detected:
            #     print('thisis4')
            for box in yolo_result_task3.boxes:
                cls_id = int(box.cls[0])
                label = class_names_task3[cls_id]
                print(label)
                cx, cy, bw, bh = box.xywhn[0].tolist()
                print(bh)
                # if x< cx*1280:
                #     turn_left = True
                # else:
                #     turn_right = True    
                if ((label == 'Car' and bh > 0.27) or (label == 'Bus' and bh > 0.36) or (label == 'Motorcycle' and bh > 0.22)):
                    print(bh)
                # if label in ['Car', 'Bus', 'Motorcycle']: 
                    # if x_center==480:
                    #     if x< cx*1280:
                    #         turn_left = True
                    #     else:
                    #         turn_right = True
                    turn_left = True      
                    print("step0") 
                    detected=True
                    action_label = label
                    speed = 0.25
                    speed_go=0.28
                    speed_boost=0.2
                    speed_2=0.12
                    steer_straight = 0.0
                    steer_left = -3.0
                    steer_right = 3.0
                    

                    if turn_left:
                        
                        avg_steer = np.mean(steering_buffer)
                        print(f"[TASK3] 평균 조향값: {avg_steer:.2f}")
                        if abs(avg_steer) > 0.3:
                            if avg_steer > 0:
                                run_motion(-1.8, 0.12, 0.2)
                            else:
                                run_motion(1.8, 0.12, 0.2)
                            run_motion(0.0, 0.12, 0.15)
                
                        # run_motion(steer_straight, speed, 0.2)   # 직진
                        run_motion(steer_left, speed, 0.83)       # 좌회전
                        run_motion(steer_straight,speed_go, 0.65)
                        run_motion(steer_right, speed, 0.91)     # 우회전
                        run_motion(steer_straight, speed_go, 1.15)
                        run_motion(steer_right, speed, 0.27)       # 우회전
                        run_motion(steer_straight,speed, 0.75)
                        # run_motion(steering*1.3, 0.15, 1.2)   # 좌회전
                        # run_motion(steer_straight, speed, 0.4) 
                        detected = False
                        task_three_clear = True
                        
                    break
                break

        elif not task_four_clear:
            height = 0.01

            # 1) 신호등 먼저 탐지 → sig_task4_clear 판단
            if not task4_sig_start:
                for box in yolo_result_task4.boxes:
                    cls_id = int(box.cls[0])
                    label = class_names_task4.get(cls_id, "Unknown")
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    height = int(y2 - y1)
                    if label in ['Left', 'Straight', 'Right'] and height > 50:
                        task4_sig_start=True
                        print('11111')
                        break
            if task4_sig_start:    
                for box in yolo_result_task4.boxes:
                    cls_id = int(box.cls[0])
                    label = class_names_task4.get(cls_id, "Unknown")
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    height = int(y2 - y1)    

                    if label in ['Red', 'Green'] and height > 50:
                        cur_sig = label
                        print('22222')
                        if label == "Red":
                            speed = 0.0
                            steering *= 0.3
                            sig_task4_clear = True
                        elif label != "Red":
                            sig_task4_clear = True
                        break  # 신호등 판단 후 종료

            # 2) 신호 확인된 경우에만 방향 인식
            if sig_task4_clear:
                for box in yolo_result_task4.boxes:
                    cls_id = int(box.cls[0])
                    label = class_names_task4.get(cls_id, "Unknown")
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    height = int(y2 - y1)
                    print(height)

                    if label in ['Left', 'Straight', 'Right'] and height > 72:
                        print(height)
                        detected = True
                        cur_sig = label
                        is_left = (label == 'Left')
                        is_right = (label == 'Right')
                        is_straight = (label == 'Straight')
                        print(f"[TASK4] Direction detected: {label}", flush=True)
                        break

            # 3) 진로 방향 판단 후 회전 실행
            if detected:
                speed = 0.18
                steer_straight = 0.0
                steer_left = -2.0
                steer_right = 2.0

                if is_left:
                    run_motion(steering*0.3, speed, 0.95)
                    run_motion(steer_left, speed, 0.4)
                    run_motion(steer_straight, speed, 0.25)
                    run_motion(steer_left, speed, 0.4)
                    run_motion(steer_straight, speed, 0.35)
                elif is_right:
                    run_motion(steering*0.3, speed, 0.95)
                    run_motion(steer_right, speed, 0.4)
                    run_motion(steer_straight, speed, 0.25)
                    run_motion(steer_right, speed, 0.4)
                    run_motion(steer_straight, speed, 0.35)
                elif is_straight:
                    run_motion(steering*0.2, 0.2, 3.5)

                task_four_clear = True

        # print(f"sig={cur_sig}", flush=True)
        if task_four_clear:
            update_vehicle_motion(0.0,0.0)
            time.sleep(10.0)                 
          
        # =================== 주행 제어 ===================
        update_vehicle_motion(steering, speed)
        
        steering_buffer.append(steering)

        print(f"📍 Pred: ({int(x)}, {int(y)}), x_err={x_error:.1f}, steer={steering:.3f}, sig={cur_sig},task1={task_one_clear}, task2={task_two_clear} ,task3={task_three_clear}, task4={task_four_clear}", flush=True)
        time.sleep(0.03333)
        

except KeyboardInterrupt:
    print("\n🛑 사용자 종료. 모터 정지.", flush=True)
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
