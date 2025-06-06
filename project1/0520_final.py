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

# ========== 제어 파라미터 ==========
MAX_STEER = 4.0
MAX_SPEED = 0.5
Kp = 0.04
Ki=0.005

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
class_names = {int(k): v.capitalize() for k, v in yolo_model.names.items()}
class_names_task3 = {int(k): v.capitalize() for k, v in yolo_model_task3.names.items()}
min_w, min_h = 0.03, 0.11

# ========== 상태 변수 ==========
switching_count = 0
# prev_sig = "Green"
# cur_sig = "Green"
task_one_stop_and_go = False
task_one_clear = True
task_two_clear = True
task_three_clear = False
task_four_clear = False

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
    left_ratio = 1.15 * max(1.0 - steer_val, 0)
    right_ratio = 1.15 * max(1.0 + steer_val, 0)

    L = clip(base_speed * left_ratio, MAX_SPEED)
    R = clip(base_speed * right_ratio, MAX_SPEED)

    if speed_val > 0:
        L, R = -L, -R

    send_control_async(-L, -R)
    print(f"[UGV] Steering={steer_val:.2f}, Speed={speed_val:.2f} → L={L:.2f}, R={R:.2f}", flush=True)

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
    fixed_speed = 0.15    

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

        if x > 580:
            x_center = 580
        elif x < 380:
            x_center = 380
        else:
            x_center = 480

        x_error = x_center - 480
        steering = clip(Kp * x_error+Ki*x_error*0.0333, MAX_STEER)

        # =================== 객체 인식 ===================
        yolo_result = yolo_model.predict(frame, conf=0.5, verbose=False)[0]
        yolo_result_task3 = yolo_model_task3.predict(frame, conf=0.5, verbose=False)[0]
        # cur_sig = "None"
        speed = fixed_speed

        # Task 1: 신호등 처리
        if not task_one_clear:
            # cur_sig = prev_sig
            for box in yolo_result.boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                cx, cy, bw, bh = box.xywhn[0].tolist()

                if label in ['Red', 'Green'] and bw > min_w and bh > min_h:
                    detected = True
                    cur_sig = label
                    break    
            #!!!!!!!!!!!!!!!!!!!!!!!!!!green만유지될경우도!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if detected and task_one_stop_and_go and cur_sig == "Green":
                task_one_clear = True
                detected = False
                cur_sig = "None"

            elif detected and cur_sig == "Red":
                speed = 0.0
                steering *= 0.3
                task_one_stop_and_go=True
            
            elif detected and cur_sig != "Red":
                    start = time.time()
                    if time.time() - start > 7.5:
                        task_one_clear = True
                        detected = False
                        cur_sig = "None"
                                


        # Task 2: 정지표지판, 느린표지판
        elif not task_two_clear:
            if action_label is not None:
                if cur_time - action_start_time < 1.5:
                    cur_sig = action_label
                    detected = True
                    if action_label == "Stop":
                        speed = 0.0
                    elif action_label == "Slow":
                        speed = 0.10
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

                    if label in ['Stop', 'Slow'] and bh > min_h*1.2:
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
            turn_left = False
            turn_right= False
            if not detected:
                for box in yolo_result_task3.boxes:
                    cls_id = int(box.cls[0])
                    label = class_names_task3[cls_id]
                    print(label)
                    cx, cy, bw, bh = box.xywhn[0].tolist()
                    # if x< cx*1280:
                    #     turn_left = True
                    # else:
                    #     turn_right = True    
                    if label in ['Car', 'Bus', 'Motorcycle'] and bh > 0.4:
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
                        speed_go=0.3
                        speed_boost=0.39
                        speed_2=0.12
                        steer_straight = 0.0
                        steer_left = -3.0
                        steer_right = 3.0

                        if turn_left:
                            print("aa")
                            # run_motion(steer_straight, speed, 0.2)   # 직진
                            run_motion(steer_left, speed, 0.9)       # 좌회전
                            run_motion(steer_straight,speed_go,0.25)
                            run_motion(steer_right, speed, 0.85)     # 우회전
                            run_motion(steer_straight, speed_go, 1.2)
                            run_motion(steer_right, speed, 0.85)       # 우회전
                            run_motion(steer_straight,speed_boost,0.35)
                            run_motion(steer_left, speed, 0.78)     # 좌회전
                            # run_motion(steer_straight, speed, 0.4) 
                            detected = False
                            task_three_clear = True

                        # elif turn_right :
                        #     print("bb")
                        #     # run_motion(steer_straight, speed, 0.2)   # 직진
                        #     run_motion(steer_right, speed, 0.8)       # 우회전
                        #     run_motion(steer_left, speed, 0.87)     # 좌회전
                        #     run_motion(steer_straight, speed_go, 1.0)
                        #     run_motion(steer_left, speed, 0.87)       # 좌회전
                        #     run_motion(steer_right, speed, 0.8)
                        #     run_motion(steer_straight, speed_2, 0.1) 
                        #     detected = False
                        #     task_three_clear = True
                            
                        break
                    break

        elif not task_four_clear:
            steering*=0.5
            if not detected:
                for box in yolo_result.boxes:
                    cls_id = int(box.cls[0])
                    label = class_names[cls_id]
                    cx, cy, bw, bh = box.xywhn[0].tolist()

                    if label in ['Left', 'Straight', 'Right'] and bw > min_w and bh > min_h:
                        detected = True
                        print(label)
                        cur_sig = label
                        print(f"sig={cur_sig}", flush=True)
                        break
                if detected and cur_sig == "Left":
                    speed = 0.18
                    steer_straight = 0.0
                    steer_left = -3.0

                    run_motion(steer_straight, speed, 0.3)   # 직진
                    run_motion(steer_left, speed, 0.9)       # 좌회전
                    run_motion(steer_straight, speed, 0.25)   # 직진
                    run_motion(steer_left, speed, 0.9)       # 좌회전
                    run_motion(steer_straight, speed, 0.1)   # 직진

                    task_four_clear = True

                elif detected and cur_sig == "Right":
                    speed = 0.18
                    steer_straight = 0.0
                    steer_right = 3.0

                    run_motion(steer_straight, speed, 0.3)   # 직진
                    run_motion(steer_right, speed, 0.9)       # 좌회전
                    run_motion(steer_straight, speed, 0.25)   # 직진
                    run_motion(steer_right, speed, 0.9)       # 좌회전
                    run_motion(steer_straight, speed, 0.1)   # 직진
                    task_four_clear = True
                
                elif detected and cur_sig == "Straight":
                    speed = 0.2
                    steer_straight = 0.0

                    run_motion(steer_straight, speed, 2.5)   # 직진
                    task_four_clear = True

        # print(f"sig={cur_sig}", flush=True)
        if task_four_clear:
            update_vehicle_motion(0.0,0.0)
            time.sleep(10.0)                 
          
        # =================== 주행 제어 ===================
        update_vehicle_motion(steering, speed)

        print(f"📍 Pred: ({int(x)}, {int(y)}), x_err={x_error:.1f}, steer={steering:.3f}, sig={cur_sig},task1={task_one_clear}, task2={task_two_clear} ,task3={task_three_clear}, task4={task_four_clear}", flush=True)
        time.sleep(0.03333)

except KeyboardInterrupt:
    print("\n🛑 사용자 종료. 모터 정지.", flush=True)
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
