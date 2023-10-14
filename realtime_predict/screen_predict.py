# 修改截取方式
# 优化了窗口捕获和推理的帧数（约30帧）
# 修改为按Esc结束（请务必使用Esc结束窗口运行，否则极容易出现“无权限访问”的问题，需要焦点在窗口上按Esc）
# 新增多线程调用优化
# 新增names显示，修改数字为names里面的名字(兼容中文)
# 新增窗口置顶和大小调整
# 新增名称/置信度，FPS显示
# forecast.py
import os
import sys
import threading
import time
from queue import Queue

import cv2
import numpy as np
import pygetwindow as gw
import win32con
import win32gui
import yaml
from mss import mss

from ultralytics import YOLO

import datetime

import json

# 从config中读取JSON数据
with open(r'E:\application\Security_monitoring_system\config\config.json', 'r') as f:
    data_config = json.load(f)

def load_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']


# 替换为你的data.yaml 里面保存有nc和names的那个，可以使用中文，推理出来的就是你names的名字，不是0,1数字了
names_path = r"E:\application\Security_monitoring_system\realtime_predict\data.yaml"
names = load_names(names_path)

# 窗口名称和大小 ，推理时允许手动拖拽窗口大小
window_name = "YOLOv8 Smoke&Fire Predict"
display_window_width = 768
display_window_height = 432
# 27 按Esc结束
exit_code = 27
text_output = ''

while_flag = False
global image_det
image_det = []
# exit_code = ord("q")  # 按q结束


def capture_screen(img_queue):
    with mss() as sct:
        monitor = sct.monitors[1]  # 这里的数字应该就是代表着显示器编号

        while True:
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_queue.put(img)
            if while_flag:
                break



class YoloThread(threading.Thread):
    def __init__(self, model, img_queue, result_queue):
        threading.Thread.__init__(self)
        self.model = model
        self.img_queue = img_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            img = self.img_queue.get()
            results = self.model.predict(source=img, conf=data_config['conf'], iou=data_config['iou'])  # 预测结果，应该包含了类、框等信息
            self.result_queue.put((img, results))
            if while_flag:
                break



def run(model, top_most=True):
    window_flag = cv2.WINDOW_NORMAL  # 标记为可调整大小的窗口
    fps_update_interval = 0.5  # 每0.5秒更新一次
    img_update_interval = 5  # 检测图片每?秒更新一次
    frame_counter = 0
    last_fps_update_time = time.time()
    last_img_update_time = time.time()

    fps = 0

    img_queue = Queue()
    result_queue = Queue()

    capture_thread = threading.Thread(target=capture_screen, args=(img_queue,))  # 截图线程
    capture_thread.daemon = True
    capture_thread.start()

    yolo_thread = YoloThread(model, img_queue, result_queue)  # 识别线程
    yolo_thread.daemon = True
    yolo_thread.start()

    # 将这两个函数放在 while True 循环之外
    cv2.namedWindow(window_name, window_flag)
    cv2.resizeWindow(window_name, display_window_width, display_window_height)

    # 初始化输出信息
    global text_output

    global while_flag


    while True:
        current_frame_time = time.time()
        img, results = result_queue.get()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, current_time, (display_window_width - 550, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 2, cv2.LINE_AA)  # 绘制时间
        for result in results:
            if len(result.boxes.xyxy) > 0:  # result.boxes.xyxy是一个列表，其中每个元素都是一个四元组，表示预测框的位置信息。
                boxes_conf = np.array(result.boxes.conf.tolist())  # result.boxes.conf是一个列表，其中每个元素都是一个浮点数，表示预测框的置信度。
                boxes_xyxy = result.boxes.xyxy.tolist()
                boxes_cls = result.boxes.cls.tolist()  # result.boxes.cls是一个列表，其中每个元素都是一个整数，表示预测框的类别。

                for i, box_xyxy in enumerate(boxes_xyxy):

                    class_name = names[int(boxes_cls[i])]  # 获取类名

                    if class_name == 'fire':
                        cv2.putText(img, 'warning:fire!', (display_window_width - 350, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 150), 4, cv2.LINE_AA)
                        # 在检测屏幕上显示文本，第一个数字是字大小，第二个是粗细
                        cv2.rectangle(img, (int(box_xyxy[0]), int(box_xyxy[1])),
                                      (int(box_xyxy[2]), int(box_xyxy[3])), (0, 0, 100), 2)  # 绘制矩形框

                        confidence_text = f"{class_name}: {boxes_conf[i]:.2f}"
                        cv2.putText(img, confidence_text, (int(box_xyxy[0]), int(box_xyxy[1]) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 150), 2, cv2.LINE_AA)

                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        text_output = text_output + '时间：' + current_time + ' 发现火焰!' + '\n'

                    elif class_name == 'smoke':
                        cv2.putText(img, 'warning:smoke!', (display_window_width - 350, 350),
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 150), 4, cv2.LINE_AA)
                        cv2.rectangle(img, (int(box_xyxy[0]), int(box_xyxy[1])),
                                      (int(box_xyxy[2]), int(box_xyxy[3])), (0, 0, 100), 2)  # 绘制矩形框
                        # 在图像上绘制一个文本框，文本框的内容由变量confidence_text给出，文本框的位置由矩形框的左上角坐标
                        # (int(box_xyxy[0]), int(box_xyxy[1]) - 20)确定。这个函数使用OpenCV库中的putText()函数来绘制文本框。
                        # 其中，cv2.FONT_HERSHEY_SIMPLEX表示文本字体，1表示文本大小，(0, 0, 150)表示文本颜色，(0, 0, 150)表示蓝色。
                        confidence_text = f"{class_name}: {boxes_conf[i]:.2f}"
                        cv2.putText(img, confidence_text, (int(box_xyxy[0]), int(box_xyxy[1]) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 150), 2, cv2.LINE_AA)

                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        text_output = text_output + '检测时间：' + current_time + ' 发现烟雾!' + '\n'

                elapsed_time_img = current_frame_time - last_img_update_time
                if elapsed_time_img >= img_update_interval:
                    # 将 BGR 格式的 frame 转换为 RGB 格式
                    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # 把 rgb_frame 转换为 numpy格式 就行了
                    numpy_frame = np.array(rgb_frame)
                    image_det.append(numpy_frame)
                    last_img_update_time = current_frame_time



        frame_counter += 1
        elapsed_time = current_frame_time - last_fps_update_time

        fps_text = f"FPS: {fps}"
        cv2.putText(img, fps_text, (display_window_width - 700, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 2, cv2.LINE_AA)  # 绘制帧率




        if elapsed_time >= fps_update_interval:  # 刷新帧率显示
            fps = int(frame_counter / elapsed_time)
            frame_counter = 0
            last_fps_update_time = current_frame_time
        # 在一个名为window_name的窗口中显示图像img。这个函数使用OpenCV库中的imshow()函数来显示图像。
        # 如果窗口不存在，则会创建一个新窗口。如果窗口已经存在，则会更新窗口中的图像。
        cv2.imshow(window_name, img)

        if top_most:
            window = gw.getWindowsWithTitle(window_name)[0]
            window_handle = window._hWnd
            win32gui.SetWindowPos(window_handle, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        if cv2.waitKey(1) == exit_code:
            cv2.destroyAllWindows()
            # os._exit(0)
            # sys.exit()
            while_flag = True
            capture_thread.join()
            yolo_thread.join()
            break


if __name__ == '__main__':
    # 是否默认窗口置顶
    top_most = True
    # 你的best.pt模型(生成模型之后千万记得替换模型，我就因为忘记替换模型还以为我数据集有问题hh)
    model = YOLO(r"E:\application\Security_monitoring_system\models\smoke_fire.pt")

    run(model, top_most)


