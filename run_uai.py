import numpy as np
import csv
import time
import sys
import argparse
import os
import random
import protocols
import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model-name',
    default='grid',
    help='type of graphical model')
parser.add_argument(
    '-alg', '--algorithms',
    nargs = '+',
    default=['bp', 'ijgp', 'mbe', 'wmbe', 'mbr', 'gbr'],
    help = 'algorithms to be tested')
parser.add_argument(
    '--seed',
    type=int,
    default=0)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
model_type='uai'

model_protocol = protocols.model_protocol_dict[model_type]
inference_protocols = [
    protocols.inference_protocol_dict[name]
    for name in args.algorithms]

size = 15
min_delta, max_delta, nb_delta = 0.0, 2.0, 9
deltas = np.linspace(min_delta, max_delta, nb_delta)
ibound = 10
nb_experiments = 1
file_name = ('delta[model={}_ibound={:d}_delta[min={:.1f}_max={:.1f}_num={:d}]].csv'.format(
    model_type, ibound, min_delta, max_delta, nb_delta))

for i in range(nb_experiments):
    model = model_protocol['generator'](args.model_name)
    true_logZ = model_protocol['true_inference'](model)
    for ip in inference_protocols:
        if ip['use_ibound']:
            alg = ip['algorithm'](model, ibound)
        else:
            alg = ip['algorithm'](model)

        tic = time.time()
        logZ = alg.run(**ip['run_args'])
        err = np.abs(true_logZ - logZ)
        toc = time.time()

        print('Alg: {:15}, Error: {:15.4f}, Time: {:15.2f}'.format(
            ip['name'], err, toc-tic))

        utils.append_to_csv(file_name, [args.model_name, ip['name'], err, toc-tic])
