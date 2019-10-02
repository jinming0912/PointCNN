from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pointfly as pf
import tensorflow as tf
# ./train_val_modelnet.sh -g 0 -x modelnet_x3_l4_aligned


def xconv(pts, fts, pts_prev, fts_prev, qrs, tag, N, K, D, P, C, C_prev, C_pts_fts, is_training, with_X_transformation, use_channel_wise, depth_multiplier,
          sorting_method=None, with_global=False, get_k='nearest', rate=1):


    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, get_k, True)
    indices = indices_dilated[:, :, ::D, :]

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    _, nn_pts_K = tf.split(nn_pts, [1, K-1], axis=-2)
    print('nn_pts_K', nn_pts_K.get_shape())
    nn_pts_K = tf.squeeze(nn_pts_K, [2]) # (N, P, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)



    if fts is None:
        nn_fts_input = nn_fts_from_pts
        nn_fts = nn_fts_input
        Chnnel_fts_input = C_pts_fts

    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts = tf.gather_nd(fts, indices, name=tag + 'nn_fts')  # (N, P, K, C)
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')
        Chnnel_fts_input = C_pts_fts + C_prev

    print('nn_fts', nn_fts.get_shape())
    print('nn_fts_input', nn_fts_input.get_shape())
    print('Chnnel_fts_input = ', Chnnel_fts_input)


    if fts_prev is None:
        #nn_pts_local = nn_pts_local
        _, nn_pts_local_prev = tf.split(nn_pts_local, [1, K-1], axis=-2)
        print("nn_pts_local_prev", nn_pts_local_prev.get_shape())
        _, nn_pts_local_K_prev = tf.split(nn_pts_local, [1, K - 1], axis=-2)
        print("nn_pts_local_K_prev", nn_pts_local_K_prev.get_shape())

    else:
        _, indices_dilated_prev = pf.knn_indices_general(qrs, pts_prev, K * D, get_k, True)
        _, indices_dilated_K_prev = pf.knn_indices_general(nn_pts_K, pts_prev, K * D, get_k, True)
        indices_prev = indices_dilated_prev[:, :, ::D, :]
        indices_K_prev = indices_dilated_K_prev[:, :, ::D, :]
        if sorting_method is not None:
            indices_prev = pf.sort_points(pts_prev, indices_prev, sorting_method)
            indices_K_prev = pf.sort_points(pts_prev, indices_K_prev, sorting_method)
        nn_pts_prev = tf.gather_nd(pts_prev, indices_prev, name=tag + 'nn_pts_prev')  # (N, P, K, 3)
        print("nn_pts_prev", nn_pts_prev.get_shape())
        nn_pts_K_prev = tf.gather_nd(pts_prev, indices_K_prev, name=tag + 'nn_pts_K_prev')  # (N, P, K, 3)
        print("nn_pts_K_prev", nn_pts_K_prev.get_shape())
        nn_pts_local_prev = tf.subtract(nn_pts_prev, nn_pts_center, name=tag + 'nn_pts_local_prev')  # (N, P, K, 3)
        print("nn_pts_local_prev", nn_pts_local_prev.get_shape())
        nn_pts_local_K_prev = tf.subtract(nn_pts_K_prev, nn_pts_center, name=tag + 'nn_pts_local_K_prev')  # (N, P, K, 3)
        print("nn_pts_local_K_prev", nn_pts_local_K_prev.get_shape())
        _, nn_pts_local_prev = tf.split(nn_pts_local_prev, [1, K-1], axis=-2)# (N, P, 1, 3)
        print("nn_pts_local_prev", nn_pts_local_prev.get_shape())
        _, nn_pts_local_K_prev = tf.split(nn_pts_local_K_prev, [1, K - 1], axis=-2)# (N, P, 1, 3)
        print("nn_pts_local_K_prev", nn_pts_local_K_prev.get_shape())



    #nn_pts_use = tf.concat([nn_pts_local, nn_fts], axis=3, name=tag + 'nn_pts_use')  # (N, P, K, C+3)
    nn_pts_local_use = tf.concat([nn_pts_local, nn_pts_local_prev, nn_pts_local_K_prev], axis=-2, name=tag + 'nn_pts_local_use')  # (N, P, K*K, C)
    print("nn_pts_local_use", nn_pts_local_use.get_shape())



    if with_X_transformation:

        if use_channel_wise:

            rate = rate


            ######################## channel-wise X-transformation #########################
            X_0 = pf.conv2d(nn_pts_local_use, K*K * Chnnel_fts_input // rate, tag + 'X_0', is_training, (1, K*K))
            X_0_KK = tf.reshape(X_0, (N, P, K*K, Chnnel_fts_input // rate), name=tag + 'X_0_KK')
            X_1 = pf.conv2d(X_0_KK, K*K * Chnnel_fts_input // rate, tag + 'X_1', is_training, (1, K*K))
            X_1_KK = tf.reshape(X_1, (N, P, K*K, Chnnel_fts_input // rate), name=tag + 'X_1_KK')
            X_2 = pf.conv2d(X_1_KK, K * Chnnel_fts_input // rate, tag + 'X_2', is_training, (1, K*K), activation=None)
            X_2_KK = tf.reshape(X_2, (N, P, K, Chnnel_fts_input // rate), name=tag + 'X_2_KK')
            X_2_KK = tf.tile(X_2_KK, multiples=[1, 1, 1, rate])
            print('X_2_KK', X_2_KK.get_shape())
            fts_X = tf.multiply(X_2_KK, nn_fts_input, name=tag + 'fts_X')
            print('use channel wise')
            print('fts_X', fts_X.get_shape())
            ###################################################################

        else:


            ######################## X-transformation #########################
            X_0 = pf.conv2d(nn_pts_local_use, K*K * K*K, tag + 'X_0', is_training, (1, K*K))
            X_0_KK = tf.reshape(X_0, (N, P, K*K, K*K), name=tag + 'X_0_KK')
            print('X_0_KK', X_0_KK.get_shape())
            X_1 = pf.conv2d(X_0_KK, K*K * K*K, tag + 'X_1', is_training, (1, K*K))
            X_1_KK = tf.reshape(X_1, (N, P, K*K, K*K), name=tag + 'X_1_KK')
            print('X_1_KK', X_1_KK.get_shape())
            X_2 = pf.conv2d(X_1_KK, K*K, tag + 'X_2', is_training, (1, K*K), activation=None)
            X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
            print('X_2_KK', X_2_KK.get_shape())
            fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
            print('fts_X', fts_X.get_shape())
            ###################################################################

            # ######################## original X-transformation #########################
            # X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))
            # X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
            # X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
            # X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
            # X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
            # X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
            # fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
            # ###################################################################

    else:
        fts_X = nn_fts_input

    fts_conv = pf.conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K))
    print('fts_conv', fts_conv.get_shape())

    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')
    print('fts_conv_3d', fts_conv_3d.get_shape())


    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d


class PointCNN:
    def __init__(self, points, features, is_training, setting):
        xconv_params = setting.xconv_params
        fc_params = setting.fc_params
        with_X_transformation = setting.with_X_transformation
        sorting_method = setting.sorting_method
        N = tf.shape(points)[0]

        alpha = tf.Variable(1e-15, name="alpha", trainable=True)

        if setting.sampling == 'fps':
            from sampling import tf_sampling

        self.layer_pts = [points]
        if features is None:
            self.layer_fts = [features]
        else:
            features = tf.reshape(features, (N, -1, setting.data_dim - 3), name='features_reshape')
            C_fts = xconv_params[0]['C'] // 2
            features_hd = pf.dense(features, C_fts, 'features_hd', is_training)
            self.layer_fts = [features_hd]

        for layer_idx, layer_param in enumerate(xconv_params):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K']
            D = layer_param['D']
            P = layer_param['P']
            C = layer_param['C']
            links = layer_param['links']
            if setting.sampling != 'random' and links:
                print('Error: flexible links are supported only when random sampling is used!')
                exit()

            # get k-nearest points
            pts = self.layer_pts[-1]
            fts = self.layer_fts[-1]
            if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']):
                qrs = self.layer_pts[-1]

            else:
                if setting.sampling == 'fps':
                    fps_indices = tf_sampling.farthest_point_sample(P, pts)
                    batch_indices = tf.tile(tf.reshape(tf.range(N), (-1, 1, 1)), (1, P, 1))
                    indices = tf.concat([batch_indices, tf.expand_dims(fps_indices,-1)], axis=-1)
                    qrs = tf.gather_nd(pts, indices, name= tag + 'qrs') # (N, P, 3)
                elif setting.sampling == 'ids':
                    indices = pf.inverse_density_sampling(pts, K, P)
                    qrs = tf.gather_nd(pts, indices)
                elif setting.sampling == 'random':
                    qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
                else:
                    print('Unknown sampling method!')
                    exit()
            self.layer_pts.append(qrs)

            if layer_idx == 0:
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
                C_prev = C_pts_fts
                pts_prev = None
                fts_prev = None
            else:
                C_prev = xconv_params[layer_idx - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
                pts_prev = self.layer_pts[-3]
                fts_prev = self.layer_fts[-2]

            with_global = (setting.with_global and layer_idx == len(xconv_params) - 1)

            use_channel_wise = True
            rate = 1

            if layer_idx>3:
                use_channel_wise = False

            # if layer_idx>3:
            #     rate = 4

            get_k = 'nearest'
            # if layer_idx >= 11:
            #     get_k = 'farthest'
            fts_xconv = xconv(pts, fts, pts_prev, fts_prev, qrs, tag, N, K, D, P, C, C_prev, C_pts_fts, is_training, with_X_transformation, use_channel_wise,
                              depth_multiplier, sorting_method, with_global, get_k, rate)
            fts_list = []
            for link in links:
                fts_from_link = self.layer_fts[link]
                if fts_from_link is not None:
                    fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1), name=tag + 'fts_slice_' + str(-link))
                    fts_list.append(fts_slice)
            if fts_list:
                fts_list.append(fts_xconv)
                self.layer_fts.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
            else:
                self.layer_fts.append(fts_xconv)

        if hasattr(setting, 'xdconv_params'):
            for layer_idx, layer_param in enumerate(setting.xdconv_params):
                tag = 'xdconv_' + str(layer_idx + 1) + '_'
                K = layer_param['K']
                D = layer_param['D']
                pts_layer_idx = layer_param['pts_layer_idx']
                qrs_layer_idx = layer_param['qrs_layer_idx']

                pts = self.layer_pts[pts_layer_idx + 1]
                fts = self.layer_fts[pts_layer_idx + 1] if layer_idx == 0 else self.layer_fts[-1]
                qrs = self.layer_pts[qrs_layer_idx + 1]
                fts_qrs = self.layer_fts[qrs_layer_idx + 1]
                P = xconv_params[qrs_layer_idx]['P']
                C = xconv_params[qrs_layer_idx]['C']
                C_prev = xconv_params[pts_layer_idx]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = 1
                fts_xdconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                                   depth_multiplier, sorting_method)
                fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
                self.layer_pts.append(qrs)
                self.layer_fts.append(fts_fuse)

        self.fc_layers = [self.layer_fts[-1]]
        self.fc_layers.append(tf.layers.dropout(self.fc_layers[-1], 0.5, training=is_training, name='fc_input_drop'))

        for layer_idx, layer_param in enumerate(fc_params):
            C = layer_param['C']
            dropout_rate = layer_param['dropout_rate']
            fc = pf.dense(self.fc_layers[-1], C, 'fc{:d}'.format(layer_idx), is_training)
            fc_drop = tf.layers.dropout(fc, dropout_rate, training=is_training, name='fc{:d}_drop'.format(layer_idx))
            self.fc_layers.append(fc_drop)
