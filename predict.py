#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
import json
import re
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import sys
from mask_rcnn import MASK_RCNN

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    mask_rcnn = MASK_RCNN()
    mode = "dir_predict"
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    test_interval   = 100
    fps_image_path  = "img/street.jpg"

    dir_origin_path = "predict_input"
    dir_save_path   = "predict_out"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = mask_rcnn.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(mask_rcnn.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = mask_rcnn.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm
        for personal_data in os.listdir(dir_origin_path):
            pattern = r'[\u4e00-\u9fa5]+'  # 匹配所有连续的中文字符
            patient_name = ''.join(re.findall(pattern, personal_data))
            patient_data_save_name = os.path.join(dir_save_path,
                                                  dir_origin_path + "_" + patient_name + ".json")
            with open(patient_data_save_name, "w", encoding="utf-8") as file:
                patient_data = {'patient_name': patient_name}
                person_data_dir = os.path.join(dir_origin_path, personal_data)
                if os.path.isdir(person_data_dir):
                    # 获取图片文件名：
                    img_names = os.listdir(person_data_dir)
                    # 创建输出病人输出文件夹名：
                    images_list = []
                    count = 0
                    for img_name in tqdm(img_names):
                        if img_name.lower().endswith(
                                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                            # 排除所有的病例图片
                            img_data = {}
                            if len(str(img_name)) >= 8:
                                img_data['img_name'] = str(img_name)

                                image_path = os.path.join(dir_origin_path, img_name)
                                image = Image.open(image_path)
                                width = image.width
                                offset = (float(width) - 512.0) / 2.0
                                if width != 512:
                                    cut_image = image.crop((offset, 0, offset + 512, 512))
                                else:
                                    cut_image = image

                                r_image, lymph_list = mask_rcnn.detect_image(cut_image, img_name)
                                if not os.path.exists(dir_save_path):
                                    os.makedirs(dir_save_path)
                                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95,
                                             subsampling=0)

                                img_data['lymph_data'] = lymph_list

                            images_list.append(img_data)
                    patient_data['data'] = images_list
                    patient_data['img_count'] = count
                    json.dump(patient_data, file, ensure_ascii=False, indent=4)
                file.close()
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
