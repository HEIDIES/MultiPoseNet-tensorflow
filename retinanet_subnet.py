import tensorflow as tf
import layer


class RETINANETSUBNET:
    def __init__(self, name, is_training, num_anchors=9):
        self.name = name
        self.is_training = is_training
        self.num_anchors = num_anchors
        self.reuse = False

    def __call__(self, res_block_c3, res_block_c4, res_block_c5):
        with tf.variable_scope(self.name):
            retina5 = layer.retina5(res_block_c5, reuse=self.reuse, is_training=self.is_training)
            retina6 = layer.retina6(res_block_c5, reuse=self.reuse, is_training=self.is_training)
            retina7 = layer.retina7(retina6, reuse=self.reuse, is_training=self.is_training)
            retina4 = layer.retina4(res_block_c4, retina5, reuse=self.reuse, is_training=self.is_training)
            retina3 = layer.retina3(res_block_c3, retina4, reuse=self.reuse, is_training=self.is_training)

        with tf.variable_scope('retina_class_subnet'):
            retina_class_subnet_output3 = layer.retina_class_subnet(retina3, num_anchors=self.num_anchors,
                                                                    reuse=self.reuse, is_training=self.is_training)
            retina_class_subnet_output4 = layer.retina_class_subnet(retina4, num_anchors=self.num_anchors,
                                                                    reuse=self.reuse, is_training=self.is_training)
            retina_class_subnet_output5 = layer.retina_class_subnet(retina5, num_anchors=self.num_anchors,
                                                                    reuse=self.reuse, is_training=self.is_training)
            retina_class_subnet_output6 = layer.retina_class_subnet(retina6, num_anchors=self.num_anchors,
                                                                    reuse=self.reuse, is_training=self.is_training)
            retina_class_subnet_output7 = layer.retina_class_subnet(retina7, num_anchors=self.num_anchors,
                                                                    reuse=self.reuse, is_training=self.is_training)

        with tf.variable_scope('retina_reg_subnet'):
            retina_bboxreg_subnet_output3 = layer.retina_bboxreg_subnet(retina3, num_anchors=self.num_anchors,
                                                                        reuse=self.reuse, is_training=self.is_training)
            retina_bboxreg_subnet_output4 = layer.retina_bboxreg_subnet(retina4, num_anchors=self.num_anchors,
                                                                        reuse=self.reuse, is_training=self.is_training)
            retina_bboxreg_subnet_output5 = layer.retina_bboxreg_subnet(retina5, num_anchors=self.num_anchors,
                                                                        reuse=self.reuse, is_training=self.is_training)
            retina_bboxreg_subnet_output6 = layer.retina_bboxreg_subnet(retina6, num_anchors=self.num_anchors,
                                                                        reuse=self.reuse, is_training=self.is_training)
            retina_bboxreg_subnet_output7 = layer.retina_bboxreg_subnet(retina7, num_anchors=self.num_anchors,
                                                                        reuse=self.reuse, is_training=self.is_training)

            self.reuse = True
            self.retina_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.class_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='retina_class_subnet')
            self.reg_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='retina_reg_subnet')
            return retina_class_subnet_output3, retina_class_subnet_output4, retina_class_subnet_output5, \
                retina_class_subnet_output6, retina_class_subnet_output7, retina_bboxreg_subnet_output3, \
                retina_bboxreg_subnet_output4, retina_bboxreg_subnet_output5, retina_bboxreg_subnet_output6, \
                retina_bboxreg_subnet_output7
