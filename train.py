import tensorflow as tf
from model import KEYPOINTSUBNET
from reader import Reader
from datetime import datetime
import os
import logging
import json_convert
import numpy as np
import math
# from tensorflow.contrib.slim import nets

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 8, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 224, 'image size, default: 256')
tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_integer('num_id2', 2, 'number of res_block_id2')
tf.flags.DEFINE_integer('num_id3', 3, 'number of res_block_id3')
tf.flags.DEFINE_integer('num_id4', 5, 'number of res_block_id4')
tf.flags.DEFINE_integer('num_id5', 2, 'number of res_block_id5')
tf.flags.DEFINE_bool('use_depth_to_space', True, 'using depth to space method or not')
tf.flags.DEFINE_integer('heatmap_channels', 14, 'the number of channels of heatmap')
tf.flags.DEFINE_string('X', 'data/tfrecords/train.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/image.tfrecords')
tf.flags.DEFINE_string('labels_file', 'data/label/keypoint_train_annotations_20170909.json',
                       'labels file for training, default: data/label/keypoint_validation_annotations_20170911.json')
tf.flags.DEFINE_string('load_model', '20180825-1954',
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_float('gaussian_sigma', 1.5, 'the variation of gaussian kernel, default: 1.0')
tf.flags.DEFINE_string('pretrained_model_checkpoints', 'pretrained_model/resnet_v2_50.ckpt',
                       'pretrained resnet_v2_50 model file, default: pretrained_model/resnet_v2_50.ckpt')
train_mode = True


def get_heatmap(image_ids, image_heights, image_widths, labels, gaussian_sigma):
    heatmap = np.zeros(shape=[FLAGS.batch_size, FLAGS.image_size // 4, FLAGS.image_size // 4,
                              FLAGS.heatmap_channels], dtype=np.float32)
    for i in range(len(image_ids)):
        label = labels[image_ids[i].decode('utf-8')]
        origin_height = image_heights[i]
        origin_width = image_widths[i]
        for key in label[1]:
            for j in range(len(label[1][key]) // 3):
                width = FLAGS.image_size // 4
                height = FLAGS.image_size // 4
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


def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    labels = json_convert.load_label(FLAGS.labels_file)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        keypoint_subnet = KEYPOINTSUBNET('keypoint_subnet', FLAGS.image_size,
                                         FLAGS.num_id2, FLAGS.num_id3, FLAGS.num_id4,
                                         FLAGS.num_id5, FLAGS.use_depth_to_space,
                                         heat_map_channels=FLAGS.heatmap_channels,
                                         learning_rate=FLAGS.learning_rate,
                                         batch_size=FLAGS.batch_size)

        res50_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
        saver_res50 = tf.train.Saver(res50_var_list)
        keypoint_subnet_loss, intermediate_loss = keypoint_subnet.model()
        keypoint_subnet_output = keypoint_subnet.out()
        optimizers, global_step = keypoint_subnet.keypoint_subnet_optimizer(keypoint_subnet_loss + intermediate_loss)

        saver = tf.train.Saver()

        reader = Reader(FLAGS.X, batch_size=FLAGS.batch_size, image_size=FLAGS.image_size)
        x, image_ids, image_heights, image_widths = reader.feed()

        starter_gaussian_sigma = FLAGS.gaussian_sigma
        end_gaussian_sigma = 0.5
        start_decay_step = 50000
        decay_steps = 100000
        gaussian_sigma = tf.where(
            tf.greater_equal(global_step, start_decay_step),
            tf.train.polynomial_decay(starter_gaussian_sigma,
                                      tf.cast(global_step, tf.int32)
                                      - start_decay_step,
                                      decay_steps,
                                      end_gaussian_sigma,
                                      power=1.0),
            starter_gaussian_sigma
        )

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    if train_mode is True:
        with graph.as_default():
            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(checkpoints_dir, graph)

        with tf.Session(graph=graph, config=config) as sess:
            if FLAGS.load_model is not None:
                checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
                step = int(meta_graph_path.split("-")[2].split(".")[0])
            else:
                sess.run(tf.global_variables_initializer())
                saver_res50.restore(sess, FLAGS.pretrained_model_checkpoints)
                step = 0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop() and step < 200000:

                    images, img_ids, img_widths, img_heights, g_sigma = sess.run([x, image_ids,
                                                                                  image_heights, image_widths,
                                                                                  gaussian_sigma])
                    heatmaps = get_heatmap(img_ids, img_heights, img_widths, labels, g_sigma)
                    _, keypoint_subnet_loss_val, summary = sess.run([optimizers, keypoint_subnet_loss, summary_op],
                                                                    feed_dict={keypoint_subnet.X: images,
                                                                               keypoint_subnet.Y: heatmaps})
                    train_writer.add_summary(summary, step)
                    train_writer.flush()
                    if (step + 1) % 100 == 0:
                        logging.info('-----------Step %d:-------------' % (sess.run(global_step)))
                        logging.info('  keypoint_subnet_loss   : {}'.format(keypoint_subnet_loss_val))
                        logging.info('  gaussian_sigma         : {}'.format(g_sigma))

                    if step % 10000 == 0:
                        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                        logging.info("Model saved in file: %s" % save_path)

                    step += 1

            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()

            except Exception as e:
                coord.request_stop(e)

            finally:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)
    else:
        with tf.Session(graph=graph, config=config) as sess:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                images, img_widths, img_heights = sess.run([x, image_heights, image_widths])
                pred = sess.run([keypoint_subnet_output],
                                feed_dict={keypoint_subnet.X: images,
                                keypoint_subnet.is_training: False})
                threshold = 0.5
                print(np.max(pred))

            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()

            except Exception as e:
                coord.request_stop(e)

            finally:
                coord.request_stop()
                coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
