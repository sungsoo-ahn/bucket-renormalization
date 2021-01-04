import sys
import os

from graphical_model.generate_model import generate_grid, generate_complete
from inference.bucket_elimination import BucketElimination
from inference.mean_field import MeanField
from inference.belief_propagation import BeliefPropagation

MF_PROTOCOL = {
    "name": "MF",
    "use_ibound": False,
    "algorithm": lambda model: MeanField(model),
    "run_args": {},
}
BP_PROTOCOL = {
    "name": "BP",
    "use_ibound": False,
    "algorithm": lambda model: BeliefPropagation(model),
    "run_args": {"max_iter": 1000, "converge_thr": 1e-5, "damp_ratio": 0.1},
}


COMPLETE_PROTOCOL = {
    "generator": lambda size, delta: generate_complete(nb_vars=size, delta=delta),
    "true_inference": lambda model: BucketElimination(model).run(),
}
GRID_PROTOCOL = {
    "generator": lambda size, delta: generate_grid(nb_vars=size ** 2, delta=delta),
    "true_inference": lambda model: BucketElimination(model).run(
        elimination_order_method="not_random"
    ),
}
UAI_PROTOCOL = {
    "generator": lambda model_name: generate_uai(model_name=model_name),
    "true_inference": lambda model: UAIInference(model).run(),
}

inference_protocol_dict = {
    "mf": MF_PROTOCOL,
    "bp": BP_PROTOCOL,
}

model_protocol_dict = {
    "complete": COMPLETE_PROTOCOL,
    "grid": GRID_PROTOCOL
    }
