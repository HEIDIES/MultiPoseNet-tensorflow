import tensorflow as tf
import layer


class RESNET50:
    def __init__(self, name, is_training, num_id2=2, num_id3=3, num_id4=5, num_id5=2,
                 norm='batch'):
        self.name = name
        self.num_id2 = num_id2
        self.num_id3 = num_id3
        self.num_id4 = num_id4
        self.num_id5 = num_id5
        self.norm = norm
        self.is_training = is_training
        self.reuse = False

    def __call__(self, ipt):
        with tf.variable_scope(self.name):
            c7s2k64 = layer.c7s2k64(ipt, reuse=self.reuse, is_training=self.is_training)
            c7s2k64 = tf.nn.max_pool(c7s2k64, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            res_c2 = layer.residual_block_conv2_x(c7s2k64, 1, reuse=self.reuse,
                                                  is_training=self.is_training)
            res_c2_block = []
            for i in range(self.num_id2):
                res_c2_block.append(layer.residual_block_id2_x(res_c2_block[-1] if i
                                                               else res_c2, i,
                                                               reuse=self.reuse,
                                                               is_training=self.is_training))
            res_c3 = layer.residual_block_conv3_x(res_c2_block[-1], 1, reuse=self.reuse,
                                                  is_training=self.is_training)
            res_c3_block = []
            for i in range(self.num_id3):
                res_c3_block.append(layer.residual_block_id3_x(res_c3_block[-1] if i
                                                               else res_c3, i,
                                                               reuse=self.reuse,
                                                               is_training=self.is_training))
            res_c4 = layer.residual_block_conv4_x(res_c3_block[-1], 1, reuse=self.reuse,
                                                  is_training=self.is_training)
            res_c4_block = []
            for i in range(self.num_id4):
                res_c4_block.append(layer.residual_block_id4_x(res_c4_block[-1] if i
                                                               else res_c4, i,
                                                               reuse=self.reuse,
                                                               is_training=self.is_training))
            res_c5 = layer.residual_block_conv5_x(res_c4_block[-1], 1, reuse=self.reuse,
                                                  is_training=self.is_training)
            res_c5_block = []
            for i in range(self.num_id5):
                res_c5_block.append(layer.residual_block_id5_x(res_c5_block[-1] if i
                                                               else res_c5, i,
                                                               reuse=self.reuse,
                                                               is_training=self.is_training))
        self.reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return res_c2_block[-1], res_c3_block[-1], res_c4_block[-1], res_c5_block[-1]
