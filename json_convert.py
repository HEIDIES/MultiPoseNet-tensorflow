import json
import numpy as np
import math
from os import scandir
import cv2


def load_label(file_path):
    f = open(file_path, encoding='utf-8')
    labels = json.load(f)
    label_dict = {}
    for label in labels:
        label_dict[label['image_id']] = [label['human_annotations'], label['keypoint_annotations']]
    return label_dict


def get_heatmap(image_ids, image_heights, image_widths, labels):
    heatmap = np.zeros(shape=[2, 256 // 4, 256 // 4,
                              14], dtype=np.float32)
    for i in range(len(image_ids)):
        label = labels[image_ids[i]]
        origin_height = image_heights[i]
        origin_width = image_widths[i]
        for key in label[1]:
            for j in range(len(label[1][key]) // 3):
                width = 256 // 4
                height = 256 // 4
                center_x = label[1][key][3 * j] * width / float(origin_width)
                center_y = label[1][key][3 * j + 1] * height / float(origin_height)
                point_status = label[1][key][3 * j + 2]
                if point_status != 3:
                    th = 1.6052
                    delta = math.sqrt(th * 2)

                    x0 = int(max(0, center_x - delta * 1))
                    y0 = int(max(0, center_y - delta * 1))

                    x1 = int(min(width, center_x + delta * 1))
                    y1 = int(min(height, center_y + delta * 1))

                    for y in range(y0, y1):
                        for x in range(x0, x1):
                            d = (x - center_x) ** 2 + (y - center_y) ** 2
                            exp = d / 2.0 / 1 / 1
                            if exp > th:
                                continue
                            heatmap[i][y][x][j] = max(heatmap[i][y][x][j], math.exp(-exp))
                            heatmap[i][y][x][j] = min(heatmap[i][y][x][j], 1.0)
    return heatmap


if __name__ == '__main__':
    label_ = load_label('data/label/keypoint_train_annotations_20170909.json')
    input_dir = 'data/train'
    i_ = 0
    imgs = []
    img_ids = []
    img_heights = []
    img_widths = []
    for img_file in scandir(input_dir):
        i_ += 1
        if img_file.name.endswith('.jpg') and img_file.is_file():
            img_ids.append(img_file.name[:-4])
            img = cv2.imread(img_file.path, cv2.IMREAD_COLOR)
            imgs.append(img)
            height_, width_, _ = img.shape
            img_heights.append(height_)
            img_widths.append(width_)

        if i_ == 2:
            break
    heat_map = get_heatmap(img_ids, img_heights, img_widths, label_)
    heat_map = heat_map.transpose([0, 3, 1, 2])
    cv2.imshow('origin image', imgs[0])
    print(heat_map.shape)
    cv2.imshow('head heat map', np.reshape(heat_map[0][12][:][:], (64, 64, 1)))
    cv2.imshow('right ankle heat map', np.reshape(heat_map[0][0][:][:], (64, 64, 1)))
    cv2.waitKey(0)
