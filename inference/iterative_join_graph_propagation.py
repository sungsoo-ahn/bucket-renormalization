import numpy as np
from copy import copy
from functools import reduce
import sys
sys.path.extend(['graphical_model/'])
from factor import Factor, product_over_, entropy
import random
import time
from belief_propagation import BeliefPropagation
def default_message_name(prefix = '_M'):
    default_message_name.cnt += 1
    return prefix + str(default_message_name.cnt)
default_message_name.cnt = 0
