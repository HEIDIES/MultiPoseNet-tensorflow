import tensorflow as tf
import layer


class DFEATURENET:
    def __init__(self, name, is_training, use_depth_to_space=True,
                 resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 heat_map_channels=15):
        self.name = name
        self.reuse = False
        self.is_training = is_training
        self.use_depth_to_space = use_depth_to_space
        self.resize_method = resize_method
        self.heat_map_channels = heat_map_channels

    def __call__(self, keypoint2, keypoint3, keypoint4, keypoint5):
        with tf.variable_scope(self.name):
            d_feature5 = layer.d_feature5(keypoint5, use_depth_to_space=self.use_depth_to_space,
                                          reuse=self.reuse, is_training=self.is_training,
                                          resize_method=self.resize_method)
            d_feature4 = layer.d_feature4(keypoint4, use_depth_to_space=self.use_depth_to_space,
                                          reuse=self.reuse, is_training=self.is_training,
                                          resize_method=self.resize_method)
            d_feature3 = layer.d_feature3(keypoint3, use_depth_to_space=self.use_depth_to_space,
                                          reuse=self.reuse, is_training=self.is_training,
                                          resize_method=self.resize_method)
            d_feature2 = layer.d_feature2(keypoint2, use_depth_to_space=self.use_depth_to_space,
                                          reuse=self.reuse, is_training=self.is_training)
            d_feature = tf.concat([d_feature2, d_feature3, d_feature4, d_feature5], axis=3)

            smooth = layer.smooth(d_feature, reuse=self.reuse, is_training=self.is_training)

            output = layer.output(smooth, heat_map_channels=self.heat_map_channels,
                                  reuse=self.reuse, is_training=self.is_training)

        self.reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
