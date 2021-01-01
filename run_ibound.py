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
parser.add_argument("-m", "--model-type", default="grid", help="type of graphical model")
parser.add_argument(
    "-alg", "--algorithms", nargs="+", default=["ijgp"], help="algorithms to be tested"
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

model_protocol = protocols.model_protocol_dict[args.model_type]
inference_protocols = [protocols.inference_protocol_dict[name] for name in args.algorithms]

size = 15
delta = 1.0
min_ibound, max_ibound, nb_ibound = 4, 10, 3
ibounds = np.linspace(min_ibound, max_ibound, nb_ibound)
nb_experiments = 91
file_name = "ibound[model={}_delta={:.1f}_ibound[min={:d}_max={:d}_num={:d}]].csv".format(
    args.model_type, delta, min_ibound, max_ibound, nb_ibound
)

for i in range(nb_experiments):
    for ibound in ibounds:
        ibound = int(ibound)
        model = model_protocol["generator"](size, delta)
        true_logZ = model_protocol["true_inference"](model)
        for ip in inference_protocols:
            if ip["use_ibound"]:
                alg = ip["algorithm"](model, ibound)
            else:
                alg = ip["algorithm"](model)

            tic = time.time()
            logZ = alg.run(**ip["run_args"])
            err = np.abs(true_logZ - logZ)
            toc = time.time()

            print("Alg: {:15}, Error: {:15.4f}, Time: {:15.2f}".format(ip["name"], err, toc - tic))

            utils.append_to_csv(file_name, [ibound, ip["name"], err, toc - tic])
