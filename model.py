import tensorflow as tf
# from resnet50 import RESNET50
from keypoint import KEYPOINTNET
from D_featureNet import DFEATURENET
import utils
from tensorflow.contrib.slim import nets

slim = tf.contrib.slim


class KEYPOINTSUBNET:
    def __init__(self, name, image_size, num_id2=2, num_id3=3, num_id4=5, num_id5=2, use_depth_to_space=True,
                 keypoint_resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 d_feature_resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 heat_map_channels=15,
                 learning_rate=2e-4,
                 batch_size=16,
                 ):
        self.name = name
        self.image_size = image_size
        self.num_id2 = num_id2
        self.num_id3 = num_id3
        self.num_id4 = num_id4
        self.num_id5 = num_id5
        self.use_depth_to_space = use_depth_to_space
        self.keypoint_resize_method = keypoint_resize_method
        self.d_feature_resize_method = d_feature_resize_method
        self.heat_map_channels = heat_map_channels
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.is_training = tf.placeholder_with_default(True, [], name='is_training')
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.image_size / 4, self.image_size / 4,
                                             self.heat_map_channels])

        # self.resnet50 = RESNET50('resnet50', is_training=self.is_training,
        #                          num_id2=self.num_id2, num_id3=self.num_id3,
        #                          num_id4=self.num_id4, num_id5=self.num_id5)

        arg_scope = nets.resnet_v2.resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            _, self.end_points = nets.resnet_v2.resnet_v2_50(self.X, num_classes=None,
                                                             is_training=True)

        self.keypoint = KEYPOINTNET('keypointnet', is_training=self.is_training,
                                    resize_method=self.keypoint_resize_method,
                                    heat_map_channels=self.heat_map_channels)

        self.d_featurenet = DFEATURENET('d_featurenet', is_training=self.is_training,
                                        use_depth_to_space=self.use_depth_to_space,
                                        resize_method=self.d_feature_resize_method,
                                        heat_map_channels=self.heat_map_channels)

    def keypoint_subnet_loss(self, logits, intermediate_logits):
        keypoint_loss = tf.reduce_sum(tf.nn.l2_loss(tf.concat(logits, axis=0) - self.Y)) / self.batch_size
        intermediate_loss = tf.reduce_sum(tf.nn.l2_loss(
            intermediate_logits - self.Y
        )) / self.batch_size
        return keypoint_loss, intermediate_loss

    def keypoint_subnet_optimizer(self, total_loss):
        def make_optimizer(loss, variables, name='Adam'):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, tf.cast(global_step, tf.int32)
                                              - start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, name=name).
                minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step, global_step
        # keypoint_subnet_var_list = [self.resnet50.var_list, self.keypoint.var_list, self.d_featurenet.var_list]
        resnet50_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
        keypoint_subnet_var_list = [resnet50_var_list, self.keypoint.var_list, self.d_featurenet.var_list]
        keypoint_subnet_optimizer, global_step = make_optimizer(total_loss, keypoint_subnet_var_list)
        with tf.control_dependencies([keypoint_subnet_optimizer]):
            return tf.no_op(name='optimizers'), global_step

    def model(self):
        # res_block_c2, res_block_c3, res_block_c4, res_block_c5 = self.resnet50(self.X)
        res_block_c2, res_block_c3, res_block_c4, res_block_c5 = \
            self.end_points['resnet_v2_50/block1/unit_2/bottleneck_v2'], \
            self.end_points['resnet_v2_50/block2/unit_3/bottleneck_v2'], \
            self.end_points['resnet_v2_50/block3/unit_4/bottleneck_v2'], \
            self.end_points['resnet_v2_50/block4']
        keypoint2, keypoint3, keypoint4, keypoint5, intermediate_output = self.keypoint(res_block_c2, res_block_c3,
                                                                                        res_block_c4, res_block_c5)
        output = self.d_featurenet(keypoint2, keypoint3, keypoint4, keypoint5)
        keypoint_subnet_loss, intermediate_loss = self.keypoint_subnet_loss(output, intermediate_output)

        tf.summary.scalar('keypoint_subnet_loss', keypoint_subnet_loss)

        tf.summary.image('origin_image', utils.batch_convert2int(self.X))
        tf.summary.image('right_ankle_ground_truth', utils.batch_convert2int(tf.reshape(
            tf.transpose(self.Y, [3, 0, 1, 2])[0], shape=[-1, self.image_size // 4, self.image_size // 4, 1])))
        tf.summary.image('head_ground_truth', utils.batch_convert2int(tf.reshape(
            tf.transpose(self.Y, [3, 0, 1, 2])[12], shape=[-1, self.image_size // 4, self.image_size // 4, 1])))
        tf.summary.image('neck_ground_truth', utils.batch_convert2int(tf.reshape(
            tf.transpose(self.Y, [3, 0, 1, 2])[13], shape=[-1, self.image_size // 4, self.image_size // 4, 1])))

        # output_show = tf.zeros_like(output)
        # output_show = tf.where(tf.greater_equal(output, 0.5), output, output_show)

        tf.summary.image('right_ankle_predict', utils.batch_convert2int(tf.reshape(
            tf.transpose(output, [3, 0, 1, 2])[0], shape=[-1, self.image_size // 4, self.image_size // 4, 1])))
        tf.summary.image('head_predict', utils.batch_convert2int(tf.reshape(
            tf.transpose(output, [3, 0, 1, 2])[12], shape=[-1, self.image_size // 4, self.image_size // 4, 1])))
        tf.summary.image('neck_predict', utils.batch_convert2int(tf.reshape(
            tf.transpose(output, [3, 0, 1, 2])[13], shape=[-1, self.image_size // 4, self.image_size // 4, 1])))

        return intermediate_loss, keypoint_subnet_loss

    def out(self):
        res_block_c2, res_block_c3, res_block_c4, res_block_c5 = \
            self.end_points['resnet_v2_50/block1/unit_2/bottleneck_v2'], \
            self.end_points['resnet_v2_50/block2/unit_3/bottleneck_v2'], \
            self.end_points['resnet_v2_50/block3/unit_4/bottleneck_v2'], \
            self.end_points['resnet_v2_50/block4']
        keypoint2, keypoint3, keypoint4, keypoint5, intermediate_output = self.keypoint(res_block_c2, res_block_c3,
                                                                                        res_block_c4, res_block_c5)
        return self.d_featurenet(keypoint2, keypoint3, keypoint4, keypoint5)

