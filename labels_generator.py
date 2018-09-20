import numpy as np
import math
import hyper_parameters
import json_convert


def get_keypoint_heatmap(image_ids, image_heights, image_widths, labels, gaussian_sigma):
    heatmap = np.zeros(shape=[hyper_parameters.FLAGS.batch_size, hyper_parameters.FLAGS.image_size // 4,
                              hyper_parameters.FLAGS.image_size // 4,
                              hyper_parameters.FLAGS.heatmap_channels], dtype=np.float32)
    for i in range(len(image_ids)):
        label = labels[image_ids[i].decode('utf-8')]
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
    for i in range(len(image_ids)):
        label = labels[image_ids[i].decode('utf-8')]
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

            for k in range(5):
                heatmap = np.zeros(shape=[hyper_parameters.FLAGS.batch_size, hyper_parameters.FLAGS.image_size //
                                          (8 * math.pow(2, k)),
                                          hyper_parameters.FLAGS.image_size // (8 * math.pow(2, k)),
                                          hyper_parameters.FLAGS.num_anhors,
                                          hyper_parameters.FLAGS.num_classes +
                                          hyper_parameters.FLAGS.bbox_dims
                                          ],
                                   dtype=np.float32)
                x = center_x / origin_width * hyper_parameters.FLAGS.image_size // (8 * math.pow(2, k))
                y = center_y / origin_height * hyper_parameters.FLAGS.image_size // (8 * math.pow(2, k))
                for j in range(hyper_parameters.FLAGS.num_anchors):
                    heatmap[i][int(y)][int(x)][j][0] = x
                    heatmap[i][int(y)][int(x)][j][1] = y
                    heatmap[i][int(y)][int(x)][j][2] = math.sqrt(w_obj / (hyper_parameters.FLAGS.image_size //
                                                                          (8 * math.pow(2, k))))
                    heatmap[i][int(y)][int(x)][j][3] = math.sqrt(h_obj / (hyper_parameters.FLAGS.image_size //
                                                                          (8 * math.pow(2, k))))

                heatmaps.append(np.reshape(heatmap, [hyper_parameters.FLAGS.batch_size,
                                                     hyper_parameters.FLAGS.image_size // (8 * math.pow(2, k)),
                                                     hyper_parameters.FLAGS.image_size // (8 * math.pow(2, k)),
                                                     hyper_parameters.FLAGS.num_anhors *
                                                     (
                                                          hyper_parameters.FLAGS.num_classes +
                                                          hyper_parameters.FLAGS.bbox_dims
                                                     )]))
    return heatmaps


if __name__ == '__main__':
    _labels = json_convert.load_label('data/label/keypoint_validation_annotations_20170911.json')
