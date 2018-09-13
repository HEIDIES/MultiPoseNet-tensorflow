import tensorflow as tf
import ops


def c7s2k64(ipt, name='c7s2k64', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, 64, 7, 3, 2, norm='batch', activation=tf.nn.relu,
                          name=name, reuse=reuse, is_training=is_training)


def residual_block_conv2_x(ipt, i, name='resblock_conv2_x', reuse=False, is_training=True):
    with tf.variable_scope(name + str(i)):
        c1s1k64 = ops.conv2d(ipt, 64, 1, 0, 1, norm='batch', activation=tf.nn.relu,
                             reuse=reuse, is_training=is_training, name='c1s1k64')
        c3s1k64 = ops.conv2d(c1s1k64, 64, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                             reuse=reuse, is_training=is_training, name='c3s1k64')
        c1s1k256 = ops.conv2d(c3s1k64, 256, 1, 0, 1, norm='batch', activation=None,
                              reuse=reuse, is_training=is_training, name='c1s1k256')
        shortcut = ops.conv2d(ipt, 256, 1, 0, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='shortcut')
        return tf.add(shortcut, c1s1k256)


def residual_block_id2_x(ipt, i, name='resblock_id2_x', reuse=False, is_training=True):
    with tf.variable_scope(name + str(i)):
        c1s1k64 = ops.conv2d(ipt, 64, 1, 0, 1, norm='batch', activation=tf.nn.relu,
                             reuse=reuse, is_training=is_training, name='c1s1k64')
        c3s1k64 = ops.conv2d(c1s1k64, 64, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                             reuse=reuse, is_training=is_training, name='c3s1k64')
        c1s1k256 = ops.conv2d(c3s1k64, 256, 1, 0, 1, norm='batch', activation=None,
                              reuse=reuse, is_training=is_training, name='c1s1k256')
        return tf.nn.relu(tf.add(ipt, c1s1k256))


def residual_block_conv3_x(ipt, i, name='resblock_conv3_x', reuse=False, is_training=True):
    with tf.variable_scope(name + str(i)):
        c1s1k128 = ops.conv2d(ipt, 128, 1, 0, 2, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k128')
        c3s1k128 = ops.conv2d(c1s1k128, 128, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k128')
        c1s1k512 = ops.conv2d(c3s1k128, 512, 1, 0, 1, norm='batch', activation=None,
                              reuse=reuse, is_training=is_training, name='c1s1k512')
        shortcut = ops.conv2d(ipt, 512, 1, 0, 2, norm='batch', activation=None,
                              reuse=reuse, is_training=is_training, name='shortcut')
        return tf.nn.relu(tf.add(shortcut, c1s1k512))


def residual_block_id3_x(ipt, i, name='resblock_id3_x', reuse=False, is_training=True):
    with tf.variable_scope(name + str(i)):
        c1s1k128 = ops.conv2d(ipt, 128, 1, 0, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k128')
        c3s1k128 = ops.conv2d(c1s1k128, 128, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k128')
        c1s1k512 = ops.conv2d(c3s1k128, 512, 1, 0, 1, norm='batch', activation=None,
                              reuse=reuse, is_training=is_training, name='c1s1k512')
        return tf.nn.relu(tf.add(ipt, c1s1k512))


def residual_block_conv4_x(ipt, i, name='resblock_conv4_x', reuse=False, is_training=True):
    with tf.variable_scope(name + str(i)):
        c1s1k256 = ops.conv2d(ipt, 256, 1, 0, 2, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k256')
        c3s1k256 = ops.conv2d(c1s1k256, 256, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256')
        c1s1k1024 = ops.conv2d(c3s1k256, 1024, 1, 0, 1, norm='batch', activation=None,
                               reuse=reuse, is_training=is_training, name='c1s1k1024')
        shortcut = ops.conv2d(ipt, 1024, 1, 0, 2, norm='batch', activation=None,
                              reuse=reuse, is_training=is_training, name='shortcut')
        return tf.nn.relu(tf.add(shortcut, c1s1k1024))


def residual_block_id4_x(ipt, i, name='resblock_id4_x', reuse=False, is_training=True):
    with tf.variable_scope(name + str(i)):
        c1s1k256 = ops.conv2d(ipt, 256, 1, 0, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k256')
        c3s1k256 = ops.conv2d(c1s1k256, 256, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256')
        c1s1k1024 = ops.conv2d(c3s1k256, 1024, 1, 0, 1, norm='batch', activation=None,
                               reuse=reuse, is_training=is_training, name='c1s1k1024')
        return tf.nn.relu(tf.add(ipt, c1s1k1024))


def residual_block_conv5_x(ipt, i, name='resblock_conv5_x', reuse=False, is_training=True):
    with tf.variable_scope(name + str(i)):
        c1s1k512 = ops.conv2d(ipt, 512, 1, 0, 2, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k512')
        c3s1k512 = ops.conv2d(c1s1k512, 512, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k512')
        c1s1k2048 = ops.conv2d(c3s1k512, 2048, 1, 0, 1, norm='batch', activation=None,
                               reuse=reuse, is_training=is_training, name='c1s1k2048')
        shortcut = ops.conv2d(ipt, 2048, 1, 0, 2, norm='batch', activation=None,
                              reuse=reuse, is_training=is_training, name='shortcut')
        return tf.nn.relu(tf.add(shortcut, c1s1k2048))


def residual_block_id5_x(ipt, i, name='resblock_id5_x', reuse=False, is_training=True):
    with tf.variable_scope(name + str(i)):
        c1s1k512 = ops.conv2d(ipt, 512, 1, 0, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k512')
        c3s1k512 = ops.conv2d(c1s1k512, 512, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k512')
        c1s1k2048 = ops.conv2d(c3s1k512, 2048, 1, 0, 1, norm='batch', activation=None,
                               reuse=reuse, is_training=is_training, name='c1s1k2048')
        return tf.nn.relu(tf.add(ipt, c1s1k2048))


def keypoint5(ipt, name='keypoint5', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, 256, 1, 0, 1, norm=None, activation=tf.nn.relu,
                          reuse=reuse, is_training=is_training, name='c1s1k256', use_bias=True,
                          kernel_initializer=None)


def keypoint4(ipt1, ipt2, name='keypoint4', reuse=False, is_training=True,
              resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    shape = ipt2.get_shape().as_list()
    with tf.variable_scope(name):
        c1s1k256 = ops.conv2d(ipt1, 256, 1, 0, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k256', use_bias=True,
                              kernel_initializer=None)
        upsample = tf.image.resize_images(ipt2, [shape[1] * 2, shape[2] * 2],
                                          method=resize_method)
        return tf.add(c1s1k256, upsample)


def keypoint3(ipt1, ipt2, name='keypoint3', reuse=False, is_training=True,
              resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    shape = ipt2.get_shape().as_list()
    with tf.variable_scope(name):
        c1s1k256 = ops.conv2d(ipt1, 256, 1, 0, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k256', use_bias=True,
                              kernel_initializer=None)
        upsample = tf.image.resize_images(ipt2, [shape[1] * 2, shape[2] * 2],
                                          method=resize_method)
        return tf.add(c1s1k256, upsample)


def keypoint2(ipt1, ipt2, name='keypoint2', reuse=False, is_training=True,
              resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    shape = ipt2.get_shape().as_list()
    with tf.variable_scope(name):
        c1s1k256 = ops.conv2d(ipt1, 256, 1, 0, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k256', use_bias=True,
                              kernel_initializer=None)
        upsample = tf.image.resize_images(ipt2, [shape[1] * 2, shape[2] * 2],
                                          method=resize_method)
        return tf.add(c1s1k256, upsample)


def intermediate(ipt, heat_map_channels, name='keypoint_intermediate', reuse=False, is_training=True):
    with tf.variable_scope(name):
        c1s1k_heat_map = ops.conv2d(ipt, heat_map_channels, 1, 0, 1, norm=None, activation=tf.nn.relu,
                                    reuse=reuse, is_training=is_training, use_bias=True,
                                    kernel_initializer=None)
        return c1s1k_heat_map


def d_feature5(ipt, use_depth_to_space=True, name='d_feature5',
               reuse=False, is_training=True,
               resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    with tf.variable_scope(name):
        if use_depth_to_space is True:
            d_to_s_1 = tf.depth_to_space(ipt, 2)
            c3s1k256 = ops.conv2d(d_to_s_1, 512, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k256_1', use_bias=True,
                                  kernel_initializer=None
                                  )
            d_to_s_2 = tf.depth_to_space(c3s1k256, 2)
            c3s1k256 = ops.conv2d(d_to_s_2, 512, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k256_2', use_bias=True,
                                  kernel_initializer=None
                                  )
            d_to_s_3 = tf.depth_to_space(c3s1k256, 2)
            return ops.conv2d(d_to_s_3, 128, 3, 1, 1, norm=None,
                              activation=tf.nn.relu, reuse=reuse,
                              is_training=is_training, name='c3s1k128', use_bias=True,
                              kernel_initializer=None
                              )
        else:
            shape = ipt.get_shape().as_list()
            c3s1k128 = ops.conv2d(ipt, 128, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k128_1', use_bias=True,
                                  kernel_initializer=None
                                  )
            c3s1k128 = ops.conv2d(c3s1k128, 128, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k128_2', use_bias=True,
                                  kernel_initializer=None
                                  )
            return tf.image.resize_images(c3s1k128, [shape[1] * 8, shape[2] * 8],
                                          resize_method)


def d_feature4(ipt, use_depth_to_space=True, name='d_feature4',
               reuse=False, is_training=True,
               resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    with tf.variable_scope(name):
        if use_depth_to_space is True:
            d_to_s_1 = tf.depth_to_space(ipt, 2)
            c3s1k256 = ops.conv2d(d_to_s_1, 512, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k256_1', use_bias=True,
                                  kernel_initializer=None
                                  )
            d_to_s_2 = tf.depth_to_space(c3s1k256, 2)
            c3s1k256 = ops.conv2d(d_to_s_2, 512, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k256_2', use_bias=True,
                                  kernel_initializer=None
                                  )
            return ops.conv2d(c3s1k256, 128, 3, 1, 1, norm=None,
                              activation=tf.nn.relu, reuse=reuse,
                              is_training=is_training, name='c3s1k128', use_bias=True,
                              kernel_initializer=None
                              )
        else:
            shape = ipt.get_shape().as_list()
            c3s1k128 = ops.conv2d(ipt, 128, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k128_1', use_bias=True,
                                  kernel_initializer=None
                                  )
            c3s1k128 = ops.conv2d(c3s1k128, 128, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k128_2', use_bias=True,
                                  kernel_initializer=None
                                  )
            return tf.image.resize_images(c3s1k128, [shape[1] * 4, shape[2] * 4],
                                          resize_method)


def d_feature3(ipt, use_depth_to_space=True, name='d_feature3',
               reuse=False, is_training=True,
               resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    with tf.variable_scope(name):
        if use_depth_to_space is True:
            d_to_s_1 = tf.depth_to_space(ipt, 2)
            c3s1k256 = ops.conv2d(d_to_s_1, 512, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k256_1', use_bias=True,
                                  kernel_initializer=None
                                  )
            c3s1k256 = ops.conv2d(c3s1k256, 512, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k256_2', use_bias=True,
                                  kernel_initializer=None
                                  )
            return ops.conv2d(c3s1k256, 128, 3, 1, 1, norm=None,
                              activation=tf.nn.relu, reuse=reuse,
                              is_training=is_training, name='c3s1k128', use_bias=True,
                              kernel_initializer=None
                              )
        else:
            shape = ipt.get_shape().as_list()
            c3s1k128 = ops.conv2d(ipt, 128, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k128_1', use_bias=True,
                                  kernel_initializer=None
                                  )
            c3s1k128 = ops.conv2d(c3s1k128, 128, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k128_2', use_bias=True,
                                  kernel_initializer=None
                                  )
            return tf.image.resize_images(c3s1k128, [shape[1] * 2, shape[2] * 2],
                                          resize_method)


def d_feature2(ipt, use_depth_to_space=True, name='d_feature2',
               reuse=False, is_training=True):
    with tf.variable_scope(name):
        if use_depth_to_space is True:
            c3s1k256 = ops.conv2d(ipt, 512, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k256_1', use_bias=True,
                                  kernel_initializer=None
                                  )
            c3s1k256 = ops.conv2d(c3s1k256, 512, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k256_2', use_bias=True,
                                  kernel_initializer=None
                                  )
            return ops.conv2d(c3s1k256, 128, 3, 1, 1, norm=None,
                              activation=tf.nn.relu, reuse=reuse,
                              is_training=is_training, name='c3s1k128', use_bias=True,
                              kernel_initializer=None
                              )
        else:
            c3s1k128 = ops.conv2d(ipt, 128, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k128_1', use_bias=True,
                                  kernel_initializer=None
                                  )
            c3s1k128 = ops.conv2d(c3s1k128, 128, 3, 1, 1, norm=None,
                                  activation=tf.nn.relu, reuse=reuse,
                                  is_training=is_training, name='c3s1k128_2', use_bias=True,
                                  kernel_initializer=None
                                  )
            return c3s1k128


def smooth(ipt, name='smooth', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, 512, 3, 1, 1, norm=None, activation=tf.nn.relu,
                          reuse=reuse, is_training=is_training, name='c3s1k512', use_bias=True,
                          kernel_initializer=None)


def output(ipt, heat_map_channels, name='d_feature_output', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, heat_map_channels, 1, 0, 1, norm=None,
                          activation=None, reuse=reuse,
                          is_training=is_training, name='c1s1kx', use_bias=True,
                          kernel_initializer=None)


def retina7(ipt, name='retina7', reuse=False, is_training=True):
    with tf.variable_scope(name):
        ipt = tf.nn.relu(ipt)
        return ops.conv2d(ipt, 256, 3, 1, 2, norm=None, activation=None,
                          reuse=reuse, is_training=is_training, name='c1s2k256', use_bias=True,
                          kernel_initializer=None)


def retina6(ipt, name='retina6', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, 256, 3, 1, 2, norm=None, activation=None,
                          reuse=reuse, is_training=is_training, name='c1s2256', use_bias=True,
                          kernel_initializer=None)


def retina5(ipt, name='retina5', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, 256, 1, 0, 1, norm=None, activation=None,
                          reuse=reuse, is_training=is_training, name='c1s1256', use_bias=True,
                          kernel_initializer=None)


def retina4(ipt1, ipt2, name='retina4', reuse=False, is_training=True,
            resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    shape = ipt2.get_shape().as_list()
    with tf.variable_scope(name):
        c1s1k256 = ops.conv2d(ipt1, 256, 1, 0, 1, norm=None, activation=None,
                              reuse=reuse, is_training=is_training, name='c1s1k256', use_bias=True,
                              kernel_initializer=None)
        upsample = tf.image.resize_images(ipt2, [shape[1] * 2, shape[2] * 2],
                                          method=resize_method)
        return tf.add(c1s1k256, upsample)


def retina3(ipt1, ipt2, name='retina3', reuse=False, is_training=True,
            resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    shape = ipt2.get_shape().as_list()
    with tf.variable_scope(name):
        c1s1k256 = ops.conv2d(ipt1, 256, 1, 0, 1, norm=None, activation=None,
                              reuse=reuse, is_training=is_training, name='c1s1k256', use_bias=True,
                              kernel_initializer=None)
        upsample = tf.image.resize_images(ipt2, [shape[1] * 2, shape[2] * 2],
                                          method=resize_method)
        return tf.add(c1s1k256, upsample)

