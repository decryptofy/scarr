# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

from .engine import Engine
from ..model_values.plaintext import PlainText
import numpy as np
import numba as nb


class NICV(Engine):

    def __init__(self, model_value=PlainText()) -> None:
        self.trace_counts = None
        self.means = None
        self.moments = None
        self.results = None
        
        super().__init__(model_value)

    def update(self, traces: np.ndarray, data: np.ndarray):
        self.internal_state_update(traces, data, self.trace_counts, self.sum, self.sum_sq)

    async def async_update(self, traces: np.ndarray, data: np.ndarray):
        self.internal_state_update(traces, data, self.trace_counts, self.sum, self.sum_sq)

    def calculate(self):

        self.sum = self.sum[self.trace_counts != 0, :]
        self.sum_sq = self.sum_sq[self.trace_counts != 0, :]
        self.trace_counts = self.trace_counts[self.trace_counts != 0]

        mean = np.sum(self.sum, axis=0) / np.sum(self.trace_counts)
        signals = (((self.sum / self.trace_counts[:, None]) - mean))**2
        signals *= (self.trace_counts / self.trace_counts.shape[0])[:, None]
        signals = np.sum(signals, axis=0)

        noises = np.sum(self.sum_sq, axis=0) / np.sum(self.trace_counts) - (mean)**2

        return signals / noises

    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def internal_state_update(traces: np.ndarray, data: np.ndarray, counts, sums, sums_sq):
        for sample in nb.prange(traces.shape[1]):
            for trace in range(traces.shape[0]):
                if sample == 0:
                    counts[data[trace]] += 1
                sums[data[trace], sample] += traces[trace, sample]
                sums_sq[data[trace], sample] += np.square(traces[trace, sample])

    def populate(self, sample_length):
        self.trace_counts = np.zeros((self.model_value.num_vals), dtype=np.uint16)
        self.sum = np.zeros((self.model_value.num_vals, sample_length), dtype=np.float32)
        self.sum_sq = np.zeros((self.model_value.num_vals, sample_length), dtype=np.float32)
