import os
import sys
from tqdm import tqdm

import cv2


def main(srcPath, desPath):
    desPath_videos = os.path.join(desPath, 'videos')
    srcPath_cmp = os.path.join(srcPath, 'compress')
    apps = os.listdir(srcPath_cmp)
    for app in apps:
        srcPath_cmp_app = os.path.join(srcPath_cmp, app)
        types = os.listdir(srcPath_cmp_app)
        for type in tqdm(types):
            srcPath_cmp_app_type = os.path.join(srcPath_cmp_app, type)
            names = os.listdir(srcPath_cmp_app_type)
            for i, name in enumerate(names):
                if i >= 10:
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


if __name__ == '__main__':
    src_path, des_path = 'F:/k400', 'F:/dataset_dir'
    main(src_path, des_path)
