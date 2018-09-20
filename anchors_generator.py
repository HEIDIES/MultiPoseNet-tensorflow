import numpy as np
import math


anchors_size = [14, 28, 56, 112, 224]


def anchors_generator(anchor_size, aspect_ratios, scales):
    anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            anchor_width = anchor_size / np.sqrt(ratio) * scale
            anchor_height = anchor_size * np.sqrt(ratio) * scale
            anchors.append([anchor_width, anchor_height])
    return anchors


def generate_anchors():
    anchors = []
    for size in anchors_size:
        anchors.append(anchors_generator(size, [0.5, 1.0, 2.0], [1, math.pow(2, 1/3.0), math.pow(2, 2/3.0)]))
    return anchors


if __name__ == '__main__':
    _anchors = generate_anchors()
    print(_anchors)
