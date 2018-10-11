import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 8, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_integer('image_size_retina', 256, 'image size, default: 256')
tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'initial learning rate for Adam, default: 0.0001')
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
tf.flags.DEFINE_string('load_model_keypoint', '20180825-1954',
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_float('gaussian_sigma', 1.5, 'the variation of gaussian kernel, default: 1.0')
tf.flags.DEFINE_string('pretrained_model_checkpoints', 'pretrained_model/resnet_v2_50.ckpt',
                       'pretrained resnet_v2_50 model file, default: pretrained_model/resnet_v2_50.ckpt')
tf.flags.DEFINE_integer('num_anchors', 9, 'the number of anchors, default: 9')
tf.flags.DEFINE_integer('bbox_dims', 4, 'the dimensions of bbox')
tf.flags.DEFINE_integer('num_classes', 1, 'the number of classes')
tf.flags.DEFINE_string('load_model_retina', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_float('learning_rate_retina', 0.01,
                      'initial learning rate for Adam, default: 0.01')
tf.flags.DEFINE_float('gamma', 2.0, 'weights, default: 2.0')
tf.flags.DEFINE_float('alpha', 0.25, 'weights, default: 0.25')
tf.flags.DEFINE_string('pretrained_keypoint_model', 'pretrained_model/keypoint_dfeatrue_res50.ckpt',
                       'pretrained model file, default: pretrained_model/keypoint_dfeatrue_res50.ckpt')
