import argparse
import random
import json
import numpy as np

import protocols
from inference.mean_field import MeanField

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="grid", type=str, help="type of graphical model")
    parser.add_argument("--model-size", default=4, type=int, help="size of graphical model")
    parser.add_argument("--model-delta", default=1.0, type=float, help="temperature parameter")
    parser.add_argument(
        "--mean_field_init_method",
        default="random",
        type=str,
        help="method to initialize mean field parameters",
    )
    parser.add_argument(
        "--max-iter", default=1000, type=int, help="maximum number of algorithm iterations"
    )
    parser.add_argument(
        "--converge-thr",
        default=1e-5,
        type=float,
        help="convergence threhold of estimated marginals for the algorithm to converge",
    )
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Generate GM
    model_protocol = protocols.model_protocol_dict[args.model_type]
    model = model_protocol["generator"](args.model_size, args.model_delta)

    # Compute the partition function using exact inference
    true_logZ = model_protocol["true_inference"](model)

    # Compute the partition function using belief propagation
    mean_field = MeanField(model, mean_field_init_method=args.mean_field_init_method)
    result = mean_field.run(
        max_iter=args.max_iter, converge_thr=args.converge_thr,
    )

    result["logZ_error"] = true_logZ - result["logZ"]
    print(result)
