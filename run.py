import argparse
import logging
import os

import configs
import utils
from datahandler import DataHandler
from dgc import DGC


def main(tag, seed, dataset):
    opts = getattr(configs, 'config_%s' % dataset)
    opts['work_dir'] = './results/%s/' % tag

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'],
                                  'checkpoints'))

    if opts['e_noise'] == 'gaussian' and opts['pz'] != 'normal':
        assert False, 'Gaussian encoders compatible only with Gaussian prior'

    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))
    data = DataHandler(opts, seed)
    model = DGC(opts, tag)
    model.train(data)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = utils.get_free_gpu(1)
    os.environ["OMP_NUM_THREADS"] = "8"

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default='mnist',
                        help='dataset [mnist/cifar10]')
    FLAGS = parser.parse_args()
    dataset_name = FLAGS.exp

    for seed in range(0, 1):
        tag = '%s_seed%02d' % (dataset_name, seed)
        main(tag, seed, dataset_name)
