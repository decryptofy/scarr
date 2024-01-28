# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
import numba as nb
from .engine import Engine


class NICV(Engine):

    def __init__(self) -> None:
        # Creating all of the necessary information containers to compute the SNR
        self.trace_counts = None
        self.means = None
        self.moments = None
        self.results = None

    def update(self, traces: np.ndarray, plaintext: np.ndarray):
        self.internal_state_update(traces, plaintext, self.trace_counts, self.sum, self.sum_sq)

    async def async_update(self, traces: np.ndarray, plaintext: np.ndarray):
        self.internal_state_update(traces, plaintext, self.trace_counts, self.sum, self.sum_sq)

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
    def internal_state_update(traces: np.ndarray, plaintext: np.ndarray, counts, sums, sums_sq):
        for sample in nb.prange(traces.shape[1]):
            for trace in range(traces.shape[0]):
                if sample == 0:
                    counts[plaintext[trace]] += 1
                sums[plaintext[trace], sample] += traces[trace, sample]
                sums_sq[plaintext[trace], sample] += np.square(traces[trace, sample])

    def populate(self, sample_length):
        # Count for each plaintext value
        self.trace_counts = np.zeros((256), dtype=np.uint16)
        # Mean value for each hex value and each sample point
        self.sum = np.zeros((256, sample_length), dtype=np.float32)
        # Moment value for each hex value and each sample point
        self.sum_sq = np.zeros((256, sample_length), dtype=np.float32)
