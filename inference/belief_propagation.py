import numpy as np
from copy import copy
from functools import reduce
import sys

from graphical_model.factor import Factor, product_over_, entropy
import random
import time

class BeliefPropagation:
    def __init__(self, model):
        self.model = model.copy()
        init_np_func = np.ones
        self.factors_adj_to_ = {
            var: self.model.get_adj_factors(var) for var in self.model.variables
        }

        self.messages = dict()
        for fac in model.factors:
            for var in fac.variables:
                self.messages[(fac, var)] = Factor.initialize_with_(
                    f"MESSAGE_{fac.name}->{var}", [var], init_np_func, model.get_cardinality_for_(var)
                )
                self.messages[(fac, var)].normalize()

        for fac in model.factors:
            for var in fac.variables:
                self.messages[(var, fac)] = Factor.initialize_with_(
                    f"MESSAGE_{var}->{fac.name}", [var], init_np_func, model.get_cardinality_for_(var)
                )
                self.messages[(var, fac)].normalize()

    def _compute_marginals_var(self, var):
        marginal = product_over_(*self._get_in_messages(var)).normalize(inplace=False)
        marginal.name = f"MARGINAL_{var}"
        return marginal

    def _compute_marginals_fac(self, fac):
        marginal = product_over_(fac, *self._get_in_messages(fac)).normalize(inplace=False)
        marginal.name = f"MARGINAL_{fac.name}"
        return marginal

    def run(self, max_iter=1000, converge_thr=1e-5, damp_ratio=0.1):
        for t in range(max_iter):
            old_messages = {key: item.copy() for key, item in self.messages.items()}
            self._update_messages(damp_ratio)
            if self._is_converged(converge_thr, self.messages, old_messages):
                break

        # Formula for computing marginals (or beliefs) from BP messages.
        self.beliefs = {}
        for var in self.model.variables:
            self.beliefs[var] = self._compute_marginals_var(var)

        for fac in self.model.factors:
            self.beliefs[fac] = self._compute_marginals_fac(fac)

        logZ = self.get_logZ()

        # Change the result into a more human-readable form.
        messages = {message.name: message.values for message in self.messages.values()}
        marginals = {belief.name: belief.values for belief in self.beliefs.values()}

        return {
            "logZ": logZ,
            "messages": messages,
            "marginals": marginals,
        }


    def get_logZ(self):
        logZ = 0.0
        for var in self.model.variables:
            logZ += (1 - self.model.degree(var)) * entropy(self.beliefs[var])

        for fac in self.model.factors:
            logZ += entropy(self.beliefs[fac], fac)

        return logZ

    def _update_messages(self, damp_ratio):
        temp_messages = dict()
        factor_order = copy(self.model.factors)
        random.shuffle(factor_order)
        for fac in factor_order:
            for var in fac.variables:
                next_message = self._compute_fac2var_message(fac, var)
                self.messages[(fac, var)] = (
                    damp_ratio * self.messages[(fac, var)] + (1 - damp_ratio) * next_message
                )
                self.messages[(fac, var)].name = f"MESSAGE_{fac.name}->{var}"

        variable_order = copy(self.model.variables)
        random.shuffle(variable_order)
        for var in variable_order:
            for fac in self.factors_adj_to_[var]:
                next_message = self._compute_var2fac_message(var, fac)
                if next_message is not None:
                    self.messages[(var, fac)] = (
                        damp_ratio * self.messages[(var, fac)] + (1 - damp_ratio) * next_message
                    )
                    self.messages[(var, fac)].name = f"MESSAGE_{var}->{fac.name}"

    def _compute_fac2var_message(self, fac, var):
        message = product_over_(fac, *[msg for msg in self._get_in_messages(fac, except_objs=[var])])
        message.marginalize_except_([var], inplace=True)
        message.normalize(inplace=True)
        return message

    def _compute_var2fac_message(self, var, fac):
        messages_to_var_except_fac = self._get_in_messages(var, except_objs=[fac])
        if messages_to_var_except_fac:
            message = product_over_(*messages_to_var_except_fac).normalize(inplace=False)
        else:
            message = None

        return message

    def _is_converged(self, converge_thr, messages, new_messages):
        for var in self.model.variables:
            blf = product_over_(
                *[messages[(fac, var)] for fac in self.factors_adj_to_[var]]
            ).normalize(inplace=False)
            new_blf = product_over_(
                *[new_messages[(fac, var)] for fac in self.factors_adj_to_[var]]
            ).normalize(inplace=False)
            if np.sum(np.abs(blf.values - new_blf.values)) > converge_thr:
                return False

        return True

    def _get_in_messages(self, obj, except_objs=[]):
        if obj in self.model.factors:
            return [self.messages[(var, obj)] for var in obj.variables if var not in except_objs]
        elif obj in self.model.variables:
            return [
                self.messages[(fac, obj)]
                for fac in self.factors_adj_to_[obj]
                if fac not in except_objs
            ]
        else:
            raise TypeError("Object {obj} not in the model.".format(obj=obj))
