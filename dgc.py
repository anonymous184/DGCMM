import logging
import math
import os
import time

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils
from datahandler import datashapes
from models import encoder, decoder
from result_logger import ResultLogger


class DGC(object):

    def __init__(self, opts, tag):
        tf.reset_default_graph()
        logging.error('Building the Tensorflow Graph')
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.opts = opts

        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data_shape = datashapes[opts['dataset']]

        self.add_inputs_placeholders()

        self.add_training_placeholders()
        sample_size = tf.shape(self.sample_points)[0]

        enc_mean, enc_sigmas = encoder(opts, inputs=self.sample_points,
                                       is_training=self.is_training, y=self.labels)

        enc_sigmas = tf.clip_by_value(enc_sigmas, -50, 50)
        self.enc_mean, self.enc_sigmas = enc_mean, enc_sigmas

        eps = tf.random_normal((sample_size, opts['zdim']),
                               0., 1., dtype=tf.float32)
        self.encoded = self.enc_mean + tf.multiply(
            eps, tf.sqrt(1e-8 + tf.exp(self.enc_sigmas)))
        # self.encoded = self.enc_mean + tf.multiply(
        #     eps, tf.exp(self.enc_sigmas / 2.))

        (self.reconstructed, self.reconstructed_logits), self.probs1 = \
            decoder(opts, noise=self.encoded,
                    is_training=self.is_training)
        self.correct_sum = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(self.probs1, axis=1), self.labels), tf.float32))
        # Decode the content of sample_noise
        (self.decoded, self.decoded_logits), _ = decoder(opts, reuse=True, noise=self.sample_noise,
                                                         is_training=self.is_training)
        # -- Objectives, losses, penalties
        self.loss_cls = self.cls_loss(self.labels, self.probs1)
        self.loss_mmd = self.mmd_penalty(self.sample_noise, self.encoded)
        self.loss_recon = self.reconstruction_loss(
            self.opts, self.sample_points, self.reconstructed)
        self.mixup_loss = self.MIXUP_loss(opts, self.encoded, self.labels)
        self.gmmpara_init()
        self.loss_mixture = self.mixture_loss(self.encoded)

        self.objective = self.loss_recon + opts['lambda_cls'] * self.loss_cls + opts['lambda_mixture'] * tf.cast(
            self.loss_mixture, dtype=tf.float32)
        self.objective_pre = self.loss_recon + opts['lambda'] * self.loss_mmd + self.loss_cls

        self.result_logger = ResultLogger(tag, opts['work_dir'], verbose=True)
        self.tag = tag

        logpxy = []
        dimY = opts['n_classes']
        N = sample_size
        S = opts['sampling_size']
        x_rep = tf.tile(self.sample_points, [S, 1, 1, 1])
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for i in range(dimY):
                y = tf.fill((N,), i)
                mu, log_sig = encoder(opts, inputs=self.sample_points, reuse=True, is_training=False, y=y)
                mu = tf.tile(mu, [S, 1])
                log_sig = tf.tile(log_sig, [S, 1])
                y = tf.tile(y, [S])
                eps2 = tf.random_normal((N * S, opts['zdim']), 0., 1., dtype=tf.float32)
                z = mu + tf.multiply(eps2, tf.sqrt(1e-8 + tf.exp(log_sig)))
                (mu_x, _), logit_y = decoder(opts, reuse=True, noise=z, is_training=False)
                logp = -tf.reduce_sum((x_rep - mu_x) ** 2, axis=[1, 2, 3])
                log_pyz = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit_y)
                posterior = tf.log(self.theta_p) - 0.5 * tf.log(2 * math.pi * self.lambda_p)
                self.u_p_1 = tf.expand_dims(self.u_p, 2)
                z_m = tf.expand_dims(tf.transpose(z), 1)
                aa = tf.square(z_m - self.u_p_1)
                self.lambda_p_1 = tf.expand_dims(self.lambda_p, 2)
                bb = aa / 2 * self.lambda_p_1
                posterior = tf.expand_dims(posterior, 2) - bb
                posterior_sum = tf.reduce_sum(tf.reduce_sum(posterior, axis=0), axis=0)
                bound = 0.5 * logp + opts['lambda_cls'] * log_pyz + opts['lambda_mixture'] * posterior_sum
                bound = tf.reshape(bound, [S, N])
                bound = self.logsumexp(bound) - tf.log(float(S))
                logpxy.append(tf.expand_dims(bound, 1))
            logpxy = tf.concat(logpxy, 1)
        y_pred = tf.nn.softmax(logpxy)

        self.eval_probs = y_pred
        self.test_a = 0.5 * logp
        self.test_b = log_pyz
        self.test_c = posterior_sum

        if opts['e_pretrain']:
            self.loss_pretrain = self.pretrain_loss()
        else:
            self.loss_pretrain = None

        self.add_optimizers()
        self.add_savers()

    def log_gaussian_prob(self, x, mu=0.0, log_sig=0.0):
        logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
                  - 0.5 * ((x - mu) / tf.exp(log_sig)) ** 2
        ind = list(range(1, len(x.get_shape().as_list())))
        return tf.reduce_sum(logprob, ind)

    def logsumexp(self, x):
        x_max = tf.reduce_max(x, 0)
        x_ = x - x_max
        tmp = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(x_), 0), 1e-20, np.inf))
        return tmp + x_max

    def add_inputs_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        data = tf.placeholder(
            tf.float32, [None] + shape, name='real_points_ph')
        label = tf.placeholder(tf.int64, shape=[None], name='label_ph')
        noise = tf.placeholder(
            tf.float32, [None] + [opts['zdim']], name='noise_ph')

        self.sample_points = data
        self.sample_noise = noise
        self.labels = label

    def add_training_placeholders(self):
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.lr_decay = decay
        self.is_training = is_training

    def pretrain_loss(self):
        opts = self.opts
        mean_pz = tf.reduce_mean(self.sample_noise, axis=0, keepdims=True)
        mean_qz = tf.reduce_mean(self.encoded, axis=0, keepdims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
        cov_pz = tf.matmul(self.sample_noise - mean_pz,
                           self.sample_noise - mean_pz, transpose_a=True)
        cov_pz /= opts['e_pretrain_sample_size'] - 1.
        cov_qz = tf.matmul(self.encoded - mean_qz,
                           self.encoded - mean_qz, transpose_a=True)
        cov_qz /= opts['e_pretrain_sample_size'] - 1.
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        return mean_loss + cov_loss

    def add_savers(self):
        saver = tf.train.Saver(max_to_keep=11)
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        if self.enc_mean is not None:
            tf.add_to_collection('encoder_mean', self.enc_mean)
            tf.add_to_collection('encoder_var', self.enc_sigmas)
        tf.add_to_collection('encoder', self.encoded)
        tf.add_to_collection('decoder', self.decoded)

        self.saver = saver

    def cls_loss(self, labels, logits):
        return tf.reduce_mean(tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)))

    def MIXUP_loss(self, opts, z_tilde, y):
        alpha = 1.0
        batch_size, z_dim = z_tilde.get_shape().as_list()

        def loss_func(z_tilde):
            lam = np.random.beta(alpha, alpha)
            index = np.random.permutation(len(z_tilde))
            mixed_z = lam * z_tilde + (1.0 - lam) * z_tilde[index]
            return mixed_z, index, lam

        mixed_z, index, lam = tf.py_func(loss_func, [z_tilde], [tf.float32, tf.int64, tf.float64])
        mixed_z.set_shape(z_tilde.get_shape())
        index.set_shape([batch_size, ])
        lam.set_shape(None)
        lam = tf.cast(lam, dtype=tf.float32)
        (_, _), pred_y = \
            decoder(opts, noise=mixed_z, is_training=self.is_training, reuse=True)

        y_a, y_b = y, tf.gather(y, index)
        soft1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_a, logits=pred_y)
        soft2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_b, logits=pred_y)
        loss = tf.reduce_sum(lam * soft1 + (1 - lam) * soft2, axis=-1)
        loss = tf.reduce_mean(loss)
        return loss

    def save_aug_data(self, x, y):
        filename = self.tag + "_aug.hdf5"
        with h5py.File(self.opts['work_dir'] + os.sep + filename, "w") as f:
            f.create_dataset("x", data=x)
            f.create_dataset("y", data=y)

    def augment_data(self, data, restore=False):
        if restore:
            self.saver.restore(self.sess,
                               tf.train.latest_checkpoint(
                                   os.path.join(self.opts['work_dir'], 'checkpoints')))
        x, y = data
        class_cnt = self.class_cnt
        x_aug_list = []
        y_aug_list = []
        batch_size = self.opts['batch_size']
        aug_num = [max(class_cnt) - class_cnt[i] for i in range(len(class_cnt))]
        for i, num in enumerate(aug_num):
            if num <= 0:
                continue
            x_c = x[y == i]
            y_c = y[y == i]
            rand_idx = np.random.choice(len(x_c), num)
            x_raw = x_c[rand_idx]
            y_aug = y_c[rand_idx]
            x_aug_batches = []
            batches_num = math.ceil(len(y_aug) / batch_size)
            for it in tqdm(range(batches_num)):
                start_idx = it * batch_size
                end_idx = start_idx + batch_size
                x_aug_batch = self.sess.run(self.reconstructed, feed_dict={self.sample_points: x_raw[start_idx:end_idx],
                                                                           self.labels: y_aug[start_idx:end_idx],
                                                                           self.is_training: False})
                x_aug_batches.append(x_aug_batch)
            x_aug = np.concatenate(x_aug_batches, axis=0)
            x_aug_list.append(x_aug)
            y_aug_list.append(y_aug)
        x_augs = np.concatenate(x_aug_list, axis=0)
        y_augs = np.concatenate(y_aug_list, axis=0)
        x = np.concatenate((x, x_augs), axis=0)
        y = np.concatenate((y, y_augs), axis=0)
        self.save_aug_data(x, y)

    def cal_dis(self, opts, z_tilde, max_iter=20):
        nx = z_tilde
        out = self.probs1
        n_class = opts['n_classes']
        py = tf.get_variable(name='py', shape=[out.shape[0], opts['n_classes']], initializer=tf.zeros_initializer())
        py.assign(tf.argmax(out, 1))
        ny = tf.argmax(out, 1)
        i_iter = tf.Variable(0, name='i', dtype=tf.int64)
        eta = tf.Variable(tf.zeros([opts['zdim'], ]))
        value_l = tf.Variable(np.inf, name='value_l')

        def cond1(out, nx, ny, py, eta, i_iter, max_iter):
            return tf.equal(py, ny) and tf.less(i_iter, max_iter)

        def body1(out, nx, ny, py, eta, i_iter, max_iter):
            grad_np = tf.gradients(out[py], nx)[0]
            ri = None
            j_iter = tf.Variable(0, name='j', dtype=tf.int64)
            r_i = tf.while_loop(cond2, body2, [grad_np, ri, value_l, py, j_iter, n_class])
            eta.assign_add(r_i)
            (_, _), out = \
                decoder(opts, noise=nx + eta, is_training=self.is_training, reuse=True)
            py = tf.argmax(out, 1)
            i_iter.assign_add(1)
            return (eta * eta).sum()

        def cond2(grad_np, ri, value_l, py, i, n_class):
            return i < n_class

        def body2(grad_np, ri, value_l, py, i, n_class):
            if tf.not_equal(i, py):
                grad_i = tf.gradients(out[0, i], nx)[0]
                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())
                if value_i < value_l:
                    ri = value_i / np.linalg.norm(wi.numpy().flatten()) * wi
            i = i + 1
            return ri

        r_i = tf.while_loop(cond1, body1, [out, nx, ny, py, eta, i_iter, max_iter])

    def mmd_penalty(self, sample_pz, sample_qz):
        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp(- distances_qz / 2. / sigma2_k)
            res1 += tf.exp(- distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp(- distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            if opts['pz'] == 'normal':
                Cbase = 2. * opts['zdim'] * sigma2_p
            elif opts['pz'] == 'sphere':
                Cbase = 2.
            elif opts['pz'] == 'uniform':
                # E ||x - y||^2 = E[sum (xi - yi)^2]
                #               = zdim E[(xi - yi)^2]
                #               = const * zdim
                Cbase = opts['zdim']
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat

    def gmmpara_init(self):
        self.theta_p = tf.get_variable("theta_p", [self.opts['n_classes']], tf.float32,
                                       tf.constant_initializer(1.0 / self.opts['n_classes']))
        self.u_p = tf.get_variable("u_p", [self.opts['zdim'], self.opts['n_classes']], tf.float32,
                                   initializer=tf.constant_initializer(0.0))
        self.lambda_p = tf.get_variable("lambda_p", [self.opts['zdim'], self.opts['n_classes']], tf.float32,
                                        initializer=tf.constant_initializer(1.0))

    def mixture_loss(self, z_tilde):
        z_mean_t = tf.transpose(tf.tile(tf.expand_dims(self.enc_mean, dim=1), [1, self.opts['n_classes'], 1]),
                                [0, 2, 1])
        z_log_var_t = tf.transpose(tf.tile(tf.expand_dims(self.enc_sigmas, dim=1), [1, self.opts['n_classes'], 1]),
                                   [0, 2, 1])
        Z = tf.transpose(tf.tile(tf.expand_dims(z_tilde, dim=1), [1, self.opts['n_classes'], 1]), [0, 2, 1])
        u_tensor3 = self.u_p
        lambda_tensor3 = self.lambda_p
        theta_tensor3 = self.theta_p
        a = tf.log(theta_tensor3) - 0.5 * tf.log(2 * math.pi * lambda_tensor3)
        b = tf.square(Z - u_tensor3)
        c = (2 * lambda_tensor3)
        p_c_z = tf.exp(tf.reduce_sum((a - b / c), axis=1)) + 1e-10
        gamma = p_c_z / tf.reduce_sum(p_c_z, axis=-1, keepdims=True)
        gamma_t = tf.tile(tf.expand_dims(gamma, dim=1), [1, self.opts['zdim'], 1])
        loss = tf.reduce_sum(0.5 * gamma_t * (self.opts['zdim'] * tf.log(math.pi * 2) + tf.log(lambda_tensor3) +
                                              tf.exp(z_log_var_t) / lambda_tensor3 + tf.square(
                    z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2))
        loss = loss - 0.5 * tf.reduce_sum(self.enc_sigmas + 1, axis=-1)
        loss = loss - tf.reduce_sum(tf.log(self.theta_p) * gamma, axis=-1) \
               + tf.reduce_sum(tf.log(gamma) * gamma, axis=-1)
        loss = tf.reduce_mean(loss)
        return loss

    def reconstruction_loss(self, opts, real, reconstr):
        if opts['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.5 * tf.reduce_mean(loss)
        elif opts['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[1, 2, 3])
            loss = 0.02 * tf.reduce_mean(loss)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return loss

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        return tf.train.AdamOptimizer(lr, beta1=opts["adam_beta1"])

    def add_optimizers(self):
        opts = self.opts
        lr = opts['lr']
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        ae_vars = encoder_vars + decoder_vars

        # Auto-encoder optimizer
        opt = self.optimizer(lr, self.lr_decay)
        self.ae_opt = opt.minimize(loss=self.objective,
                                   var_list=encoder_vars + decoder_vars)

        # Encoder optimizer
        if opts['e_pretrain']:
            opt = self.optimizer(lr)
            self.pretrain_opt = opt.minimize(loss=self.loss_pretrain,
                                             var_list=encoder_vars)
        else:
            self.pretrain_opt = None
        if opts['LVO']:
            self.lvo_opt = opt.minimize(loss=self.objective, var_list=encoder_vars)

    def get_z_dist(self, data):
        opts = self.opts
        covariances = []
        means = []
        for c in range(opts['n_classes']):
            imgs = data.data[data.labels == c]
            labels = data.labels[data.labels == c]
            batch_size = 128
            num_c = imgs.shape[0]
            latent_np = self.sess.run(self.encoded,
                                      feed_dict={self.sample_points: imgs[0:batch_size],
                                                 self.labels: labels[0:batch_size],
                                                 self.is_training: False})
            for i in range(1, num_c // batch_size):
                latent_ele = self.sess.run(self.encoded,
                                           feed_dict={self.sample_points: imgs[(i * batch_size):((i + 1) * batch_size)],
                                                      self.labels: labels[(i * batch_size):((i + 1) * batch_size)],
                                                      self.is_training: False})
                latent_np = np.concatenate((latent_np, latent_ele), axis=0)
            covariances.append(np.cov(np.transpose(latent_np)))
            means.append(np.mean(latent_np, axis=0))

        covariances = np.array(covariances)
        means = np.array(means)

        cfname = "{}covariances.npy".format(opts['work_dir'])
        mfname = "{}means.npy".format(opts['work_dir'])
        # print("saving multivariate: ", cfname, mfname)
        np.save(cfname, covariances)
        np.save(mfname, means)

        return means, covariances

    def sample_pz(self, num=100, z_dist=None, labels=None):
        opts = self.opts
        noise = None
        distr = opts['pz']
        if z_dist is None:
            if distr == 'uniform':
                noise = np.random.uniform(
                    -1, 1, [num, opts["zdim"]]).astype(np.float32)
            elif distr in ('normal', 'sphere'):
                mean = np.zeros(opts["zdim"])
                cov = np.identity(opts["zdim"])
                noise = np.random.multivariate_normal(
                    mean, cov, num).astype(np.float32)
                if distr == 'sphere':
                    noise = noise / np.sqrt(
                        np.sum(noise * noise, axis=1))[:, np.newaxis]
            return opts['pz_scale'] * noise
        else:
            assert labels is not None
            means, covariances = z_dist
            noise = np.array([
                np.random.multivariate_normal(means[e], covariances[e])
                for e in labels
            ])
            return noise

    def pre_train(self, data):
        opts = self.opts
        batches_num = data.num_points // opts['batch_size']
        self.num_pics = opts['plot_num_pics']
        decay = 1.
        for epoch in range(2):
            # Update learning rate if necessary
            for it in range(batches_num):
                start_idx = it * opts['batch_size']
                end_idx = start_idx + opts['batch_size']
                batch_images = data.data[start_idx:end_idx].astype(np.float)
                batch_labels = data.labels[start_idx:end_idx]
                batch_noise = self.sample_pz(num=opts['batch_size'], labels=batch_labels)
                feed_d = {
                    self.sample_points: batch_images,
                    self.sample_noise: batch_noise,
                    self.labels: batch_labels,
                    self.lr_decay: decay,
                    self.is_training: True}
                [_, loss, loss_rec, loss_cls, train_prob] = self.sess.run(
                    [self.ae_opt,
                     self.objective_pre,
                     self.loss_recon,
                     self.loss_cls,
                     self.probs1],
                    feed_dict=feed_d)
        z_dist = self.get_z_dist(data)
        return z_dist

    def pretrain_encoder(self, data):
        opts = self.opts
        steps_max = 200
        batch_size = opts['e_pretrain_sample_size']
        for step in range(steps_max):
            train_size = data.num_points
            data_ids = np.random.choice(train_size, min(train_size, batch_size),
                                        replace=False)
            batch_images = data.data[data_ids].astype(np.float)
            batch_labels = data.labels[data_ids].astype(np.int64)
            batch_noise = self.sample_pz(batch_size)

            [_, loss_pretrain] = self.sess.run(
                [self.pretrain_opt,
                 self.loss_pretrain],
                feed_dict={self.sample_points: batch_images,
                           self.labels: batch_labels,
                           self.sample_noise: batch_noise,
                           self.is_training: True})

            if opts['verbose']:
                logging.error('Step %d/%d, loss=%f' % (
                    step, steps_max, loss_pretrain))

            if loss_pretrain < 0.1:
                break

    def augment_batch(self, x, y):
        class_cnt = self.class_cnt

        max_class_cnt = max(class_cnt)
        n_classes = len(class_cnt)
        x_aug_list = []
        y_aug_list = []
        aug_rate = self.opts['aug_rate']
        if aug_rate <= 0:
            return x, y
        aug_nums = [aug_rate * (max_class_cnt - class_cnt[i]) for i in range(n_classes)]
        rep_nums = [aug_num / class_cnt[i] for i, aug_num in enumerate(aug_nums)]
        for i in range(n_classes):
            idx = (y == i)
            if rep_nums[i] <= 0.:
                x_aug_list.append(x[idx])
                y_aug_list.append(y[idx])
                continue
            n_c = np.count_nonzero(idx)
            if n_c == 0:
                continue
            x_aug_list.append(
                np.repeat(x[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
            y_aug_list.append(
                np.repeat(y[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
        if len(x_aug_list) == 0:
            return x, y
        x_aug = np.concatenate(x_aug_list, axis=0)
        y_aug = np.concatenate(y_aug_list, axis=0)
        return x_aug, y_aug

    def train(self, data):
        opts = self.opts
        self.class_cnt = [np.count_nonzero(data.labels == n) for n in range(opts['n_classes'])]
        if opts['verbose']:
            logging.error(opts)
        losses = []
        losses_rec = []
        losses_match = []
        losses_cls = []

        batches_num = math.ceil(data.num_points / opts['batch_size'])
        self.num_pics = opts['plot_num_pics']
        self.sess.run(tf.global_variables_initializer())

        if opts['e_pretrain']:
            logging.error('Pretraining the encoder')
            self.pretrain_encoder(data)
            logging.error('Pretraining the encoder done.')

        self.start_time = time.time()
        counter = 0
        decay = 1.
        wait = 0
        z_dist = self.pre_train(data)
        for epoch in range(opts["epoch_num"]):
            # Update learning rate if necessary
            start_time = time.time()
            if opts['lr_schedule'] == "manual":
                if epoch == 30:
                    decay = decay / 2.
                if epoch == 50:
                    decay = decay / 5.
                if epoch == 100:
                    decay = decay / 10.
            elif opts['lr_schedule'] == "manual_smooth":
                enum = opts['epoch_num']
                decay_t = np.exp(np.log(100.) / enum)
                decay = decay / decay_t

            elif opts['lr_schedule'] != "plateau":
                assert type(opts['lr_schedule']) == float
                decay = 1.0 * 10 ** (-epoch / float(opts['lr_schedule']))

            # Save the model
            if epoch > 0 and epoch % opts['save_every_epoch'] == 0:
                self.saver.save(self.sess,
                                os.path.join(opts['work_dir'],
                                             'checkpoints',
                                             'trained'),
                                global_step=counter)

            acc_total = 0.
            loss_total = 0.

            z_list = []
            y_list = []
            mu_list = []
            logsigma_list = []

            for it in tqdm(range(batches_num)):
                start_idx = it * opts['batch_size']
                end_idx = start_idx + opts['batch_size']
                batch_images = data.data[start_idx:end_idx]
                batch_labels = data.labels[start_idx:end_idx]
                orig_batch_labels = batch_labels
                orig_batch_images = batch_images
                if opts['augment_z'] is True:
                    batch_images, batch_labels = self.augment_batch(batch_images, batch_labels)
                train_size = len(batch_labels)
                # print(train_size, len(orig_batch_labels))
                batch_noise = self.sample_pz(len(batch_images), z_dist=z_dist, labels=batch_labels)
                if opts['LVO'] is True:
                    _ = self.sess.run(self.lvo_opt, feed_dict={self.sample_points: batch_images,
                                                               self.sample_noise: batch_noise,
                                                               self.labels: batch_labels,
                                                               self.lr_decay: decay,
                                                               self.is_training: True})

                feed_d = {
                    self.sample_points: batch_images,
                    self.sample_noise: batch_noise,
                    self.labels: batch_labels,
                    self.lr_decay: decay,
                    self.is_training: True}

                (_, loss, loss_rec, loss_cls, loss_match, correct, theta_p_final, u_p_final, lambda_p_final, mu,
                 logsigma) = self.sess.run(
                    [self.ae_opt,
                     self.objective,
                     self.loss_recon,
                     self.loss_cls,
                     self.loss_mixture,
                     self.correct_sum,
                     self.theta_p,
                     self.u_p,
                     self.lambda_p,
                     self.enc_mean,
                     self.enc_sigmas
                     ],
                    feed_dict=feed_d)
                acc_total += correct / train_size

                loss_total += loss

                if opts['lr_schedule'] == "plateau":
                    if epoch >= 30:
                        if loss < min(losses[-20 * batches_num:]):
                            wait = 0
                        else:
                            wait += 1
                        if wait > 10 * batches_num:
                            decay = max(decay / 1.4, 1e-6)
                            logging.error('Reduction in lr: %f' % decay)
                            wait = 0

                feed_d = {
                    self.sample_points: orig_batch_images,
                    # self.sample_noise: batch_noise,
                    # self.labels: batch_labels,
                    self.is_training: False}

                z_final = self.sess.run(
                    self.encoded,
                    feed_dict=feed_d)

                # print('z_final',z_final.shape)

                losses.append(loss)
                losses_rec.append(loss_rec)
                losses_match.append(loss_match)
                losses_cls.append(loss_cls)

                counter += 1

                if epoch >= 0 and epoch % opts['save_every_epoch'] == 0:
                    z_list.append(z_final)
                    y_list.append(orig_batch_labels)
                    mu_list.append(mu)
                    logsigma_list.append(logsigma)
                    # train_prob_list.append(train_prob)

            if epoch >= 0 and epoch % opts['save_every_epoch'] == 0:
                mus = np.concatenate(mu_list, axis=0)
                logsigmas = np.concatenate(logsigma_list, axis=0)
                # print('epoch-calculating zs ys', epoch)
                zs = np.concatenate(z_list, axis=0)
                ys = np.concatenate(y_list, axis=0)
                self.result_logger.save_latent_code_new(epoch, zs, ys, mus, logsigmas, theta_p_final, u_p_final,
                                                        lambda_p_final)

            # Print debug info
            now = time.time()
            # Auto-encoding test images
            [loss_rec_test, loss_cls_test] = self.sess.run(
                [self.loss_recon, self.loss_cls],
                feed_dict={self.sample_points: data.test_data[:self.num_pics],
                           self.labels: data.test_labels[:self.num_pics],
                           self.is_training: False})

            debug_str = 'EPOCH: %d/%d, BATCH/SEC:%.2f' % (
                epoch + 1, opts['epoch_num'],
                float(counter) / (now - self.start_time))
            debug_str += ' (TOTAL_LOSS=%.5f, RECON_LOSS=%.5f, ' \
                         'MATCH_LOSS=%.5f, ' \
                         'CLS_LOSS=%.5f, ' \
                         'RECON_LOSS_TEST=%.5f, ' \
                         'CLS_LOSS_TEST=%.5f, ' % (
                             losses[-1], losses_rec[-1],
                             losses_match[-1], losses_cls[-1], loss_rec_test, loss_cls_test)
            logging.error(debug_str)

            training_acc = acc_total / batches_num
            avg_loss = loss_total / batches_num
            self.result_logger.add_training_metrics(avg_loss, training_acc, time.time() - start_time)

            # if (self.opts['eval_strategy'] == 1 and (epoch + 1) % 5 == 0) or self.opts['eval_strategy'] == 2 and (
            #         (0 < epoch <= 20) or (epoch > 20 and epoch % 3 == 0)):
            self.evaluate(data, epoch)

            if epoch > 0:  # and epoch % 10 == 0:
                self.saver.save(self.sess,
                                os.path.join(opts['work_dir'],
                                             'checkpoints',
                                             'trained-final'),
                                global_step=epoch)
            self.viz_img(data, epoch)

        self.result_logger.save_metrics()
        # For FID
        # self.augment_data((data.data, data.labels))

    def evaluate(self, data, epoch):
        batch_size = self.opts['batch_size'] // 10
        batches_num = math.ceil(len(data.test_data) / batch_size)
        probs = []
        start_time = time.time()
        for it in tqdm(range(batches_num)):
            start_idx = it * batch_size
            end_idx = start_idx + batch_size
            [prob, tst_a, tst_b, tst_c] = self.sess.run(
                [self.eval_probs, self.test_a, self.test_b, self.test_c],
                feed_dict={self.sample_points: data.test_data[start_idx:end_idx],
                           self.is_training: False})
            probs.append(prob)
            # if it==1:
            # print('tst', tst_b, tst_c)
        probs = np.concatenate(probs, axis=0)
        predicts = np.argmax(probs, axis=-1)
        self.result_logger.save_prediction(epoch, data.test_labels, predicts, probs, time.time() - start_time)
        self.result_logger.save_metrics()

    def viz_img(self, data, epoch):
        x = data.data
        y = data.labels
        n_classes = self.opts['n_classes']
        batch_size = self.opts['batch_size']
        x_aug_list = []
        # y_aug_list = []
        for i in range(n_classes):
            x_c = x[y == i]
            y_c = y[y == i]
            rand_idx = np.random.choice(len(x_c), 50)
            x_raw = x_c[rand_idx]
            y_aug = y_c[rand_idx]
            x_aug_batches = []
            batches_num = math.ceil(len(x_raw) / batch_size)
            for it in (range(batches_num)):
                start_idx = it * batch_size
                end_idx = start_idx + batch_size
                x_aug_batch = self.sess.run(self.reconstructed, feed_dict={self.sample_points: x_raw[start_idx:end_idx],
                                                                           self.labels: y_aug[start_idx:end_idx],
                                                                           self.is_training: False})
                x_aug_batches.append(x_aug_batch)
            x_aug = np.concatenate(x_aug_batches, axis=0)
            x_aug_list.append(x_aug)
            # y_aug_list.append(y_aug)
        x_aug = np.concatenate(x_aug_list, axis=0)

        import torch
        from torchvision.utils import save_image
        filename = os.path.join(self.opts['work_dir'], "epoch%d.png" % epoch)
        save_image(torch.from_numpy(x_aug).permute(0, 3, 1, 2), filename, nrow=n_classes, padding=0)
        # y_aug = np.concatenate(y_aug_list, axis=0)
