UAI_PATH = "./graphical_model/UAI/"
import numpy as np


class UAIInference:
    def __init__(self, model):
        self.model = model

    def run(self):
        with open(UAI_PATH + self.model.name + ".uai.PR") as f:
            a = f.readlines()
        content = [c.strip() for c in a]
        true_logZ = float(content[1]) * np.log(10)

        return true_logZ
