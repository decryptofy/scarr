# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
import numba as nb
from .engine import Engine


class SNR(Engine):

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

        self.sum = self.sum[self.trace_counts > 0, :]
        self.sum_sq = self.sum_sq[self.trace_counts > 0, :]
        self.trace_counts = self.trace_counts[self.trace_counts > 0]

        means = self.sum / self.trace_counts[:, None]
        signals = np.var(means, axis=0)

        variances = (self.sum_sq / self.trace_counts[:, None]) - (means**2)
        variances[variances < 0] = 0
        noises = np.mean(variances, axis=0)
        noises[noises == 0] = 1

        return signals / noises

    @staticmethod
    @nb.njit(parallel=True, cache=True, nogil=True)
    def internal_state_update(traces: np.ndarray, plaintext: np.ndarray, counts, sums, sums_sq):
        for sample in nb.prange(traces.shape[1]):
            local_sums = np.empty(256, dtype=np.float64)
            local_sums_sq = np.empty(256, dtype=np.float64)
            local_counts = np.empty(256, dtype=np.uint32)
            local_sums[:] = 0.
            local_sums_sq[:] = 0.
            local_counts[:] = 0
            for trace in range(traces.shape[0]):
                if sample == 0:
                    local_counts[plaintext[trace]] += 1
                local_sums[plaintext[trace]] += traces[trace, sample]
                local_sums_sq[plaintext[trace]] += traces[trace, sample]**2

            sums[:, sample] += local_sums
            sums_sq[:, sample] += local_sums_sq
            counts += local_counts

    def populate(self, sample_length):
        # Count for each plaintext value
        self.trace_counts = np.zeros((256), dtype=np.uint32)
        # Mean value for each hex value and each sample point
        self.sum = np.zeros((256, sample_length), dtype=np.float64)
        # Moment value for each hex value and each sample point
        self.sum_sq = np.zeros((256, sample_length), dtype=np.float64)


class SNR_Evaluator(SNR):

    @staticmethod
    def _run(self, container, tile_x, tile_y, byte):
        for batch in container.get_batches_by_byte(tile_x, tile_y, byte):
            profiled_text = np.squeeze(np.bitwise_xor(batch[0], batch[1]))
            self.update(batch[-1], profiled_text)

        return tile_x, tile_y, byte, self._get_result()
