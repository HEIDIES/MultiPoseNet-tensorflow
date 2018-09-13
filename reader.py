import tensorflow as tf
import utils
import matplotlib.pyplot as plt
import json_convert


class Reader:
    def __init__(self, tfrecords_file, image_size=256, min_queue_examples=400, batch_size=4,
                 num_threads=12, name=''):
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.name = name
        self.reader = tf.TFRecordReader()

    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/train': tf.FixedLenFeature([], tf.string),
                    'image_id/train': tf.FixedLenFeature([], tf.string),
                    'image_height/train': tf.FixedLenFeature([], tf.int64),
                    'image_width/train': tf.FixedLenFeature([], tf.int64)
                })

            image_train_buffer = features['image/train']
            image_ids = features['image_id/train']
            image_heights = features['image_height/train']
            image_widths = features['image_width/train']
            image_train = tf.image.decode_jpeg(image_train_buffer, channels=3)
            image_train = self._image_preprocess(image_train)
            image_train, image_ids, image_heights, image_widths = tf.train.shuffle_batch(
                [image_train, image_ids, image_heights, image_widths], batch_size=self.batch_size,
                num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=self.min_queue_examples
            )

            # tf.summary.image('_input', train)
        return image_train, image_ids, image_heights, image_widths

    def _image_preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
        image = utils.convert2float(image)
        image.set_shape([self.image_size, self.image_size, 3])
        return image


def test_reader():
    train_file_1 = 'data/tfrecords/train.tfrecords'
    label_ = json_convert.load_label('data/label/keypoint_validation_annotations_20170911.json')

    with tf.Graph().as_default():
        reader1 = Reader(train_file_1, batch_size=1)
        image_train, image_ids, image_heights, image_widths = reader1.feed()
        image_train = tf.squeeze(image_train, 0)
        image = utils.convert2int(image_train)
        # image_label = tf.squeeze(image_label, 0)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop() and step < 1:
                train, image_id, image_height, image_width = sess.run([image, image_ids,
                                                                       image_heights,
                                                                       image_widths])
                print(label_[image_id[0].decode('utf-8')])
                print('image_id : ', image_id[0].decode('utf-8'))
                print('height : ', image_height)
                print('width : ', image_width)
                f, a = plt.subplots(1, 1)
                # for i in range(1):
                a.imshow(train)
                plt.show()
                step += 1
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    test_reader()
