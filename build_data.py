import tensorflow as tf
import random
import os
from os import scandir
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('train_input_dir', 'data/train',
                       'train input directory, default: data/train')

tf.flags.DEFINE_string('output_dir', 'data/tfrecords/train.tfrecords',
                       'output directory, default: data/tfrecords/train.tfrecords')


def data_reader(train_input_dir, shuffle):
    """Read images from input_dir then shuffle them
    Args:
        train_input_dir: string, path of input train dir, e.g., /path/to/dir
    Returns:
        file_paths: list of strings
    """
    train_file_paths = []
    train_image_ids = []
    train_image_widths = []
    train_image_heights = []

    for img_file in scandir(train_input_dir):
        if img_file.name.endswith('.jpg') and img_file.is_file():
            train_file_paths.append(img_file.path)
            train_image_ids.append(img_file.name[:-4])
            img = cv2.imread(img_file.path, cv2.IMREAD_COLOR)
            height, width, _ = img.shape
            train_image_heights.append(height)
            train_image_widths.append(width)

    if shuffle is True:
        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable.
        shuffled_index = list(range(len(train_file_paths)))
        random.seed(12345)
        random.shuffle(shuffled_index)

        train_file_paths = [train_file_paths[i] for i in shuffled_index]
        train_image_ids = [train_image_ids[i] for i in shuffled_index]
        train_image_heights = [train_image_heights[i] for i in shuffled_index]
        train_image_widths = [train_image_widths[i] for i in shuffled_index]

    return train_file_paths, train_image_ids, train_image_heights, train_image_widths


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _strs_feature(value):
    """Wrapper for inserting strs features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(train_data, train_id, train_height, train_width):
    """Build an Example proto for an example.
    Args:
        train_data: string, path to an image file, e.g., '/path/to/example.JPG'
    Returns:
        Example proto
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/train': _bytes_feature(train_data),
        'image_id/train': _strs_feature(train_id),
        'image_height/train': _int64_feature(train_height),
        'image_width/train': _int64_feature(train_width)
        }))
    return example


def data_writer(train_input_dir, output_file):
    """Write data to tfrecords
    """
    train_file_paths, train_image_ids, train_image_heights, train_image_widths = data_reader(train_input_dir, True)

    # create tfrecords dir if not exists
    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir)
    except os.error:
        pass

    images_num = len(train_file_paths)

    # dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(images_num):
        train_file_path = train_file_paths[i]
        train_image_id = train_image_ids[i]
        train_image_height = train_image_heights[i]
        train_image_width = train_image_widths[i]
        train_image_id = bytes(train_image_id, 'utf-8')

        with tf.gfile.FastGFile(train_file_path, 'rb') as f:
            train_data = f.read()

        example = _convert_to_example(train_data, train_image_id, train_image_height, train_image_width)
        writer.write(example.SerializeToString())

        if (i + 1) % 1000 == 0:
            print("Processed {}/{}.".format(i + 1, images_num))
    print("Done.")
    writer.close()


def main(unused_argv):
    print("Convert train and label to tfrecords...")
    data_writer(FLAGS.train_input_dir, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
