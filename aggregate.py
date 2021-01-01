import os
import sys
import glob
import utils
import numpy as np
from collections import OrderedDict

dir_name = "./results"
agg_dir_name = "./agg_results"
file_names = glob.glob("{}/*.csv".format(dir_name))
file_names = [file_name.split("/")[-1] for file_name in file_names]
# file_names = [''.join(file_name.split('.')[:-1]) for file_name in file_names]
alg_names = ["MBR", "GBR", "MBE", "WMBE", "BP", "MF", "IJGP"]
for file_name in file_names:
    print(file_name)
    err_file_name = "err_" + file_name
    time_file_name = "time_" + file_name
    contents = utils.read_csv(file_name, dir_name=dir_name)
    err_dict, time_dict = dict(), dict()
    for c in contents:
        key1, key2, err, time = c
        if key1 not in err_dict:
            err_dict[key1] = OrderedDict()
            time_dict[key1] = OrderedDict()

        if key2 not in err_dict[key1]:
            err_dict[key1][key2] = []
            time_dict[key1][key2] = []

        err_dict[key1][key2].append(float(err))
        time_dict[key1][key2].append(float(time))

    for key1 in sorted(err_dict.keys()):
        print("Key 1: {}".format(key1, list(err_dict[key1].keys())))
        avg_values, avg_times = [], []
        for key2 in sorted(err_dict[key1].keys(), key=alg_names.index):
            avg_values.append(np.mean(err_dict[key1][key2]) / np.log(10))
            avg_times.append(np.mean(time_dict[key1][key2]))
            print(
                "Key 2: {:15}, Error: {:15.4f}, Time: {:15.2f}, Num: {:15}".format(
                    key2,
                    np.mean(err_dict[key1][key2]),
                    np.mean(time_dict[key1][key2]),
                    len(err_dict[key1][key2]),
                )
            )
        utils.append_to_csv(err_file_name, avg_values, dir_name=agg_dir_name)
        utils.append_to_csv(time_file_name, avg_times, dir_name=agg_dir_name)
