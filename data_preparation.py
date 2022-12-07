import os
import sys
from tqdm import tqdm

import cv2


def main(srcPath, desPath):
    desPath_videos = os.path.join(desPath, 'videos')
    srcPath_cmp = os.path.join(srcPath, 'compress')
    apps = os.listdir(srcPath_cmp)

    train_list, val_list, test_list = [], [], []
    for app in apps:
        srcPath_cmp_app = os.path.join(srcPath_cmp, app)
        types = os.listdir(srcPath_cmp_app)
        types = sorted(types)
        for i, type in tqdm(enumerate(types)):
            srcPath_cmp_app_type = os.path.join(srcPath_cmp_app, type)
            names = os.listdir(srcPath_cmp_app_type)
            for j, name in enumerate(names):
                if j >= 10:
                    break
                srcPath_cmp_app_type_name = os.path.join(srcPath_cmp_app_type, name)
                desPath_videos_name = os.path.join(desPath_videos, name)
                os.makedirs(desPath_videos_name, exist_ok=True)
                video = cv2.VideoCapture(srcPath_cmp_app_type_name)
                ret, frame = video.read()
                cnt = 1
                while ret:
                    if frame is None:
                        break
                    save_path = os.path.join(desPath_videos_name, '{:05d}.jpg'.format(cnt))
                    cnt += 1
                    cv2.imwrite(save_path, frame)
                    ret, frame = video.read()

                # folder_path, start_frame, end_frame, label_id
                num_frames = len(os.listdir(desPath_videos_name))
                folder_path = os.path.join('videos', name)
                start_frame, end_frame = str(1), str(num_frames)
                label_id = str(i)
                str_img = folder_path + ' ' + start_frame + ' ' + end_frame + ' ' + label_id
                if app == 'train_256':
                    train_list.append(str_img)
                if app == 'val_256':
                    val_list.append(str_img)
    train_note = open(os.path.join(desPath, 'train.txt'), 'w')
    for s in train_list:
        train_note.write(s + '\n')
    train_note.close()

    val_note = open(os.path.join(desPath, 'val.txt'), 'w')
    for s in val_list:
        val_note.write(s + '\n')
    val_note.close()

    test_note = open(os.path.join(desPath, 'test.txt'), 'w')
    for s in test_list:
        test_note.write(s + '\n')
    test_note.close()


if __name__ == '__main__':
    src_path, des_path = 'F:/k400', 'F:/dataset_dir'
    main(src_path, des_path)
