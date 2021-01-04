import sys
from copy import copy
import numpy as np
import time

from graphical_model.factor import Factor, product_over_, entropy

class MeanField:
    def __init__(self, model, **kwargs):
        self.model = model.copy()

        mean_field_init_method = kwargs.get("mean_field_init_method")
        if mean_field_init_method == "random":
            init_np_func = np.ones
        elif mean_field_init_method == "uniform":
            init_np_func = np.random.random
        else:
            init_np_func = np.ones

        self.mean_fields = {}
        for var in self.model.variables:
            self.mean_fields[var] = Factor.initialize_with_(
                default_message_name(), [var], init_np_func, model.get_cardinality_for_(var)
            )
            self.mean_fields[var].normalize()

        self.messages = dict()

    def run(self, max_iter=1000, converge_thr=1e-2):
        for t in range(max_iter):
            old_mean_field = {var: copy(self.mean_fields[var]) for var in self.model.variables}
            self._update_mean_fields()
            if self._is_converged(self.mean_fields, old_mean_field, converge_thr):
                break

        logZ = self.get_logZ()

        # Change the result into a more human-readable form.
        messages = dict()
        for key, message in self.messages.items():
            if key[0] in self.model.variables:
                key = f"{key[0]}->{key[1].name}"
            else:
                key = f"{key[0].name}->{key[1]}"

            messages[key] = message.values

        marginals = dict()
        for key, mean_field in self.mean_fields.items():
            if key not in self.model.variables:
                key = key.name

            marginals[key] = mean_field.values

        return {
            "logZ": logZ,
            "messages": messages,
            "marginals": marginals,
        }

    def get_logZ(self):
        logZ = 0
        for var in self.model.variables:
            logZ += entropy(self.mean_fields[var])

        for fac in self.model.factors:
            m = product_over_(*[self.mean_fields[var] for var in fac.variables])
            index_to_keep = m.values != 0
            logZ += np.sum(m.values[index_to_keep] * fac.log_values[index_to_keep])

        return logZ

    def _update_mean_fields(self):
        variable_order = np.random.permutation(self.model.variables)
        for var in variable_order:
            for fac in self.model.get_adj_factors(var):
                self.messages[(fac, var)] = self._compute_message(fac, var)

            next_mean_field = Factor.full_like_(self.mean_fields[var], 0.0)
            for fac in self.model.get_adj_factors(var):
                next_mean_field = next_mean_field + self.messages[(fac, var)]

            self.mean_fields[var] = next_mean_field.exp(inplace=False).normalize(inplace=False)
            self.mean_fields[var].log_values = np.nan_to_num(self.mean_fields[var].log_values)

    def _compute_message(self, fac, var):
        message = Factor(
            name=f"{fac.name}->{var}",
            variables=[var],
            values=np.ones(self.model.get_cardinality_for_(var)),
        )
        message = product_over_(
            message, *[self.mean_fields[var1] for var1 in fac.variables if var1 != var]
        )
        message.log_values = fac.log_values * message.values
        message.marginalize_except_([var])
        return message

    def _is_converged(self, mean_field, old_mean_field, converge_thr):
        for var in self.model.variables:
            if np.sum(np.abs(mean_field[var].values - old_mean_field[var].values)) > converge_thr:
                return False
        return True
