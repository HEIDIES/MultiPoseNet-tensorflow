import numpy as np
import math
import hyper_parameters
from os import scandir
import json_convert
import cv2


def get_keypoint_heatmap(image_ids, image_heights, image_widths, labels, gaussian_sigma):
    heatmap = np.zeros(shape=[hyper_parameters.FLAGS.batch_size, hyper_parameters.FLAGS.image_size // 4,
                              hyper_parameters.FLAGS.image_size // 4,
                              hyper_parameters.FLAGS.heatmap_channels], dtype=np.float32)
    for i in range(len(image_ids)):
        label = labels[image_ids[i]]
        origin_height = image_heights[i]
        origin_width = image_widths[i]
        for key in label[1]:
            for j in range(len(label[1][key]) // 3):
                width = hyper_parameters.FLAGS.image_size // 4
                height = hyper_parameters.FLAGS.image_size // 4
                center_x = int(label[1][key][3 * j] * width / float(origin_height))
                center_y = int(label[1][key][3 * j + 1] * height / float(origin_width))
                point_status = label[1][key][3 * j + 2]
                if point_status != 3:
                    th = 1.6052
                    delta = math.sqrt(th * 2)

                    x0 = int(max(0, center_x - delta * gaussian_sigma))
                    y0 = int(max(0, center_y - delta * gaussian_sigma))

                    x1 = int(min(width, center_x + delta * gaussian_sigma))
                    y1 = int(min(height, center_y + delta * gaussian_sigma))

                    for y in range(y0, y1):
                        for x in range(x0, x1):
                            d = (x - center_x) ** 2 + (y - center_y) ** 2
                            exp = d / 2.0 / gaussian_sigma / gaussian_sigma
                            if exp > th:
                                continue
                            heatmap[i][y][x][j] = max(heatmap[i][y][x][j], math.exp(-exp))
                            heatmap[i][y][x][j] = min(heatmap[i][y][x][j], 1.0)
    return heatmap


def get_detector_heatmap(image_ids, image_heights, image_widths, labels):
    heatmaps = []
    for k in range(5):
        heatmap = np.zeros(shape=[hyper_parameters.FLAGS.batch_size, hyper_parameters.FLAGS.image_size //
                                          int(8 * (2 ** k)),
                                          hyper_parameters.FLAGS.image_size // int(8 * (2 ** k)),
                                          hyper_parameters.FLAGS.num_anchors,
                                          hyper_parameters.FLAGS.num_classes +
                                          hyper_parameters.FLAGS.bbox_dims
                                          ],
                                   dtype=np.float32)
        for i in range(len(image_ids)):
            label = labels[image_ids[i]]
            origin_height = image_heights[i]
            origin_width = image_widths[i]
            for key in label[0]:
                up_left_x = label[0][key][0]
                up_left_y = label[0][key][1]
                down_right_x = label[0][key][2]
                down_right_y = label[0][key][3]
                center_x = (up_left_x + down_right_x) / 2.0
                center_y = (up_left_y + down_right_y) / 2.0
                w_obj = down_right_x - up_left_x
                h_obj = down_right_y - up_left_y
                # print('frame id : %d' %i)
                # print('up_left_point : (%02f, ' %up_left_x, '%02f)\n' %up_left_y, 
                #  'down_right_point : (%02f, ' %down_right_x, '%02f)\n' %down_right_y, 
                #  'center_point : (%02f, ' %center_x, '%02f)\n' %center_y, 
                #  'obj_width : %02f\n' %w_obj, 
                #  'obj_height : %02f\n' %h_obj)

                # print(k)
                
                x = center_x / origin_width * (hyper_parameters.FLAGS.image_size // (8 * (2 ** k)))
                y = center_y / origin_height * (hyper_parameters.FLAGS.image_size // (8 * (2 ** k)))
                # print(x, y)
                for j in range(hyper_parameters.FLAGS.num_anchors):
                    heatmap[i][int(y)][int(x)][j][0] = float(x)
                    heatmap[i][int(y)][int(x)][j][1] = float(y)
                    heatmap[i][int(y)][int(x)][j][2] = math.sqrt(w_obj / origin_width)
                    heatmap[i][int(y)][int(x)][j][3] = math.sqrt(h_obj / origin_height)
                    heatmap[i][int(y)][int(x)][j][4] = 1.0

        heatmaps.append(np.reshape(heatmap, [hyper_parameters.FLAGS.batch_size,
                                             hyper_parameters.FLAGS.image_size // int(8 * (2 ** k)),
                                             hyper_parameters.FLAGS.image_size // int(8 * (2 ** k)),
                                             hyper_parameters.FLAGS.num_anchors *
                                             (
                                                  hyper_parameters.FLAGS.num_classes +
                                                  hyper_parameters.FLAGS.bbox_dims
                                             )]))
    return heatmaps


test_mode = 'detector'


if __name__ == '__main__':
    _labels = json_convert.load_label('data/label/keypoint_train_annotations_20170909.json')
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

        if i_ == 3:
            break
    if test_mode == 'keypoint': 
        heat_map = get_keypoint_heatmap(img_ids, img_heights, img_widths, _labels, hyper_parameters.FLAGS.gaussian_sigma)
        print(heat_map.shape)
    elif test_mode == 'detector':
        heat_maps = get_detector_heatmap(img_ids, img_heights, img_widths, _labels)
        b = 0
        i = 0
        for img, width, height in zip(imgs, img_widths, img_heights):
            # print(width, height)
            heat_map = np.reshape(heat_maps[b][i], [hyper_parameters.FLAGS.image_size // int(8 * (2 ** b)),
                                             hyper_parameters.FLAGS.image_size // int(8 * (2 ** b)),
                                             hyper_parameters.FLAGS.num_anchors,
                                             hyper_parameters.FLAGS.num_classes +
                                             hyper_parameters.FLAGS.bbox_dims])
            _wh = np.square(heat_map[:, :, 0, 2 : 4]) * np.reshape([width, height], [1, 1, 1, 2])
            current_size = hyper_parameters.FLAGS.image_size // (8 * (2 ** b))
            _centers = heat_map[:, :, 0, 0 : 2] * np.reshape([width, height], [1, 1, 1, 2]) / \
                           np.reshape([current_size, current_size], [1, 1, 1, 2])
            print(current_size)
            _up_left, _down_right = _centers - (_wh * 0.5), _centers + (_wh * 0.5)
            _up_left = np.squeeze(_up_left, 0)
            _down_right = np.squeeze(_down_right, 0)
            _confs = heat_map[:, :, 0, 4]
            rows, cols = _confs.shape
            for j in range(cols):
                for k in range(rows):
                    if _confs[j][k] >= 0.5:
                        # print('center : (%d, ' %_centers[0][j][k][0], '%d)' %_centers[0][j][k][1])
                        # print(_up_left[j][k], _down_right[j][k])
                        cv2.rectangle(img, (int(_up_left[j][k][0]), int(_up_left[j][k][1])), 
                           (int(_down_right[j][k][0]), int(_down_right[j][k][1])), (0, 0, 255), thickness=2)
            # cv2.rectangle(im, (10, 10), (110, 110), (0, 0, 255), thickness=2)
            i += 1
            cv2.imshow('image' + str(i), img)    
    cv2.waitKey(0)
