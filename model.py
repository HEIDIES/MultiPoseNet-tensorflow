import tensorflow as tf
# from resnet50 import RESNET50
from keypoint import KEYPOINTNET
from D_featureNet import DFEATURENET
import utils
from tensorflow.contrib.slim import nets
import numpy as np
from retinanet_subnet import RETINANETSUBNET

slim = tf.contrib.slim


class KEYPOINTSUBNET:
    def __init__(self, name, image_size, num_id2=2, num_id3=3, num_id4=5, num_id5=2, use_depth_to_space=True,
                 keypoint_resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 d_feature_resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 heat_map_channels=15,
                 learning_rate=2e-4,
                 batch_size=16
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


class DETECTERSUBNET:
    def __init__(self, name, image_size, anchors,
                 batch_size=16,
                 num_anchors=9,
                 learning_rate=0.01,
                 gamma=2.0,
                 alpha=0.25
                 ):
        self.name = name
        self.image_size = image_size
        self.anchors = anchors
        self.batch_size = batch_size
        self.num_anchors = num_anchors
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.alpha = alpha
        self.is_training = tf.placeholder_with_default(True, [], name='is_training')
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])

        self.Y_7 = tf.placeholder(tf.float32, [self.batch_size, self.image_size // 128, self.image_size // 128,
                                               self.num_anchors * 5])
        self.Y_6 = tf.placeholder(tf.float32, [self.batch_size, self.image_size // 64, self.image_size // 64,
                                               self.num_anchors * 5])
        self.Y_5 = tf.placeholder(tf.float32, [self.batch_size, self.image_size // 32, self.image_size // 32,
                                               self.num_anchors * 5])
        self.Y_4 = tf.placeholder(tf.float32, [self.batch_size, self.image_size // 16, self.image_size // 16,
                                               self.num_anchors * 5])
        self.Y_3 = tf.placeholder(tf.float32, [self.batch_size, self.image_size // 8, self.image_size // 8,
                                               self.num_anchors * 5])

        arg_scope = nets.resnet_v2.resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            _, self.end_points = nets.resnet_v2.resnet_v2_50(self.X, num_classes=None,
                                                             is_training=True)

        self.retina = RETINANETSUBNET('retinanet', is_training=self.is_training,
                                      num_anchors=self.num_anchors)

    def compute_detector_subnet_regression_loss(self, bboxes_predictions, labels, output_size=(32, 32), idx=1):
        w, h = output_size
        b = len(self.anchors[idx])
        anchors = tf.constant(self.anchors[idx], dtype=tf.float32)
        anchors = tf.reshape(anchors, [1, 1, b, 2])
        labels = tf.reshape(labels, [-1, h * w, b, 5])
        _coords = labels[:, :, :, 0: 4]
        _confs = labels[:, :, :, 4]

        c_x, c_y = list(range(w)), list(range(h))
        bs = list(range(self.batch_size))
        na = list(range(b))
        c_x, c_y = tf.meshgrid(c_x, c_y)
        c_x, _ = tf.meshgrid(c_x, bs)
        c_x, _ = tf.meshgrid(c_x, na)
        c_y, _ = tf.meshgrid(c_y, bs)
        c_y, _ = tf.meshgrid(c_y, na)
        c_x = tf.transpose(tf.reshape(c_x, [-1, na, w * h]), [0, 2, 1])
        c_y = tf.transpose(tf.reshape(c_y, [-1, na, w * h]), [0, 2, 1])
        center_grid = tf.stack([c_x, c_y], axis=3)

        _wh = tf.pow(_coords[:, :, :, 2: 4], 2) * np.reshape([w, h], [1, 1, 1, 2])
        _areas = _wh[:, :, :, 0] * _wh[:, :, :, 1]
        _centers = _coords[:, :, :, 0: 2]
        _up_left, _down_right = _centers - (_wh * 0.5), _centers + (_wh * 0.5)
        # truths = tf.concat([_coords, tf.expand_dims(_confs, -1)], axis=3)

        bboxes_predictions = tf.reshape(bboxes_predictions, [-1, h, w, b, 4])
        coords = tf.reshape(bboxes_predictions[:, :, :, :, 0: 4], [-1, h * w, b, 4])
        coords_xy = tf.nn.sigmoid(coords[:, :, :, 0: 2])
        coords_wh = tf.sqrt(tf.exp(coords[:, :, :, 2: 4]) * anchors / np.reshape([w, h], [1, 1, 1, 2]))
        coords = tf.concat([coords_xy, coords_wh], axis=3)

        # preds = tf.concat([coords, confs], axis=3)

        wh = tf.pow(coords[:, :, :, 2: 4], 2) * np.reshape([w, h], [1, 1, 1, 2])
        areas = wh[:, :, :, 0] * wh[:, :, :, 1]
        centers = coords[:, :, :, 0: 2] + center_grid
        up_left, down_right = centers - (wh * 0.5), centers + (wh * 0.5)

        inter_upleft = tf.maximum(up_left, _up_left)
        inter_downright = tf.minimum(down_right, _down_right)
        inter_wh = tf.maximum(inter_downright - inter_upleft, 0.0)
        intersects = inter_wh[:, :, :, 0] * inter_wh[:, :, : 1]
        ious = tf.truediv(intersects, areas + _areas - intersects)

        best_iou_mask = tf.equal(ious, tf.reduce_max(ious, axis=2, keep_dims=True))
        best_iou_mask = tf.cast(best_iou_mask, tf.float32)
        mask = best_iou_mask * tf.squeeze(_confs, -1)
        mask = tf.expand_dims(mask, -1)

        coors_loss = tf.square(coords - _coords) * mask
        return coors_loss

    def compute_detector_subnet_classification_loss(self, confs_predictions, labels, output_size=(32, 32), idx=1):
        w, h = output_size

        b = len(self.anchors[idx])
        labels = tf.reshape(labels, [-1, h * w, b, 5])
        _confs = labels[:, :, :, 4]

        confs_predictions = tf.reshape(confs_predictions, [-1, h, w, b, 4])
        confs = tf.nn.sigmoid(confs_predictions)
        confs = tf.reshape(confs, [-1, h * w, b, 1])

        confs_w = tf.where(tf.equal(_confs, tf.ones_like(_confs)),
                           self.alpha * tf.pow(tf.ones_like(confs) - confs, self.gamma),
                           (1 - self.alpha) * tf.pow(confs, self.gamma))
        confs_loss = -tf.reduce_mean(
            tf.reduce_sum(tf.multiply(tf.where(tf.equal(_confs, tf.ones_like(_confs)), tf.log(confs),
                                               tf.log(tf.ones_like(confs) - confs)), confs_w), axis=[1, 2, 3]))
        return confs_loss

    def detector_subnet_loss(self, bboxes_predictions, confs_predictions):
        total_reg_loss = 0.0
        total_confs_loss = 0.0

        labels = [self.Y_3, self.Y_4, self.Y_5, self.Y_6, self.Y_7]

        for i, bbox in enumerate(bboxes_predictions):
            shape = bbox.get_shape().as_list()
            label, conf = labels[i], confs_predictions[i]
            total_reg_loss += self.compute_detector_subnet_regression_loss(bbox, label,
                                                                           output_size=(shape[1], shape[2]), idx = i)
            total_confs_loss += self.compute_detector_subnet_classification_loss(conf, label,
                                                                                 output_size=(shape[1], shape[2]), idx = i)
        return total_reg_loss, total_confs_loss

    def retina_subnet_optimizer(self, reg_loss, conf_loss):
        def make_optimizer(loss, variables, name='Adam'):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 50000
            decay_steps = 50000
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
            return learning_step

        retina_class_subnet_var_list = [self.retina.class_var_list, self.retina.retina_var_list]
        retina_reg_subnet_var_list = [self.retina.reg_var_list, self.retina.retina_var_list]
        retina_class_subnet_optimizer = make_optimizer(conf_loss, retina_class_subnet_var_list)
        retina_reg_subnet_optimizer = make_optimizer(reg_loss, retina_reg_subnet_var_list)
        with tf.control_dependencies([retina_class_subnet_optimizer, retina_reg_subnet_optimizer]):
            return tf.no_op(name='optimizers')

    def model(self):
        res_block_c3, res_block_c4, res_block_c5 = \
            self.end_points['resnet_v2_50/block2/unit_3/bottleneck_v2'], \
            self.end_points['resnet_v2_50/block3/unit_4/bottleneck_v2'], \
            self.end_points['resnet_v2_50/block4']

        retina_class_subnet_output3, retina_class_subnet_output4, retina_class_subnet_output5, \
            retina_class_subnet_output6, retina_class_subnet_output7, retina_bboxreg_subnet_output3, \
            retina_bboxreg_subnet_output4, retina_bboxreg_subnet_output5, retina_bboxreg_subnet_output6, \
            retina_bboxreg_subnet_output7 = self.retina(res_block_c3, res_block_c4, res_block_c5)

        bboxes_predictions = [retina_bboxreg_subnet_output3, retina_bboxreg_subnet_output4,
                              retina_bboxreg_subnet_output5, retina_bboxreg_subnet_output6,
                              retina_bboxreg_subnet_output7]
        confs_predictions = [retina_class_subnet_output3, retina_bboxreg_subnet_output4,
                             retina_class_subnet_output5, retina_class_subnet_output6,
                             retina_class_subnet_output7]
        reg_loss, confs_loss = self.detector_subnet_loss(bboxes_predictions, confs_predictions)

        return reg_loss, confs_loss
