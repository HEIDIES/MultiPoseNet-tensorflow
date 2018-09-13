import tensorflow as tf
import layer


class KEYPOINTNET:
    def __init__(self, name, is_training, resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 heat_map_channels=15):
        self.name = name
        self.is_training = is_training
        self.reuse = False
        self.resize_method = resize_method
        self.heat_map_channels = heat_map_channels

    def __call__(self, res_block_c2, res_block_c3, res_block_c4, res_block_c5):
        with tf.variable_scope(self.name):
            keypoint5 = layer.keypoint5(res_block_c5, reuse=self.reuse, is_training=self.is_training)
            keypoint4 = layer.keypoint4(res_block_c4, keypoint5, reuse=self.reuse,
                                        is_training=self.is_training, resize_method=self.resize_method)
            keypoint3 = layer.keypoint3(res_block_c3, keypoint4, reuse=self.reuse,
                                        is_training=self.is_training, resize_method=self.resize_method)
            keypoint2 = layer.keypoint2(res_block_c2, keypoint3, reuse=self.reuse,
                                        is_training=self.is_training, resize_method=self.resize_method)

            shape5 = keypoint5.get_shape().as_list()
            shape4 = keypoint4.get_shape().as_list()
            shape3 = keypoint3.get_shape().as_list()

            intermediate_output = tf.concat(
                [
                    tf.image.resize_images(keypoint5, [shape5[1] * 8, shape5[2] * 8],
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                    tf.image.resize_images(keypoint4, [shape4[1] * 4, shape4[2] * 4],
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                    tf.image.resize_images(keypoint3, [shape3[1] * 2, shape3[2] * 2],
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                    keypoint2
                ],
                axis=3
            )

            intermediate_output = layer.intermediate(intermediate_output, self.heat_map_channels, reuse=self.reuse,
                                                     is_training=self.is_training)

        self.reuse = True
        # self.var_list_keypoint5 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/keypoint5')
        # self.var_list_keypoint4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/keypoint4')
        # self.var_list_keypoint3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/keypoint3')
        # self.var_list_keypoint2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/keypoint2')
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return keypoint2, keypoint3, keypoint4, keypoint5, intermediate_output
