# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.


from .engine import Engine
from ..model_values.model_value import ModelValue
from ..model_values.model_bits import ModelBits
import numpy as np
from multiprocessing.pool import Pool
import os


class LRA(Engine):
    def __init__(self, model_value: ModelValue, convergence_step=None, bias=True) -> None:
        self.convergence_step = convergence_step
        self.R2s = None
        self.betas = None
        self.results = []
        
        super().__init__(ModelBits(model_value, bias))

    def get_R2s(self):
        return self.R2s

    def get_betas(self):
        return self.betas

    def get_results(self):
        return self.results

    def populate(self, container):
        model_vals = self.model_value.model.num_vals
        num_steps = -(container.data.traces_length // -self.convergence_step) if self.convergence_step else 1
        
        # R2s and Beta values, Results
        self.R2s = np.zeros((len(container.model_positions),
                             num_steps,
                             model_vals,
                             container.sample_length), dtype=np.float64)
        self.betas = np.zeros((len(container.model_positions),
                               model_vals,
                               self.model_value.num_bits,
                               container.sample_length), dtype=np.float64)
        self.results = [[] for _ in range(len(container.tiles))]

    def update(self, traces: np.ndarray, plaintext: np.ndarray, group_n, group_mean, group_M):
        num_traces, num_samples = traces.shape

        for i in range(num_traces):
            t = traces[i]
            g = int(plaintext[i])
            n = group_n[g]
            n1 = n + 1
            delta = t - group_mean[g]
            group_mean[g] += delta / n1
            group_M[g] += delta * (t - group_mean[g])
            group_n[g] = n1

        return group_n, group_mean, group_M

    def calculate(self, grouped_n, grouped_mean, grouped_sumsqdiff, model):
        # Using sums of sq. differences instead of co/var eliminates divisions that would cancel out in the end
        def _ungrouped_stats(grouped_mean, grouped_sumsqdiff, group_n):
            ungrouped_mean  = np.average(grouped_mean, axis=0, weights=group_n)
            between = np.sum((grouped_mean - ungrouped_mean) ** 2 * group_n[..., None], axis=0)
            within = np.sum(grouped_sumsqdiff, axis=0)
            ungrouped_sumsqdiff = between + within
            return ungrouped_mean, ungrouped_sumsqdiff

        num_traces, trace_length = np.sum(grouped_n), grouped_sumsqdiff.shape[1]
        ungrouped_mean, ungrouped_sumsqdiff = _ungrouped_stats(grouped_mean, grouped_sumsqdiff, grouped_n)
        
        mod_mean = np.average(model, axis=1, weights=grouped_n)
        model_diff = model - mod_mean
        model_sumsqdiff = ((model_diff.swapaxes(-1, -2) * grouped_n) @ model_diff)
        mod_invcov = np.linalg.inv(model_sumsqdiff)
        signal_diff = grouped_mean - ungrouped_mean
        combined_sumsqdiff = ((model_diff.swapaxes(-1, -2) * grouped_n) @ signal_diff)
        
        betas = np.matmul(mod_invcov, combined_sumsqdiff)
        r2s = np.sum(np.divide((combined_sumsqdiff * betas), ungrouped_sumsqdiff), axis=1)

        return r2s, betas

    def find_candidate(self, R2s):
        aggregated = np.sum(R2s, axis=1)
        winning_candidate = np.argmax(aggregated)
        return winning_candidate

    def run(self, container):
        self.populate(container)

        with Pool(processes=int(os.cpu_count()/2)) as pool:
            workload = []
            for tile in container.tiles:
                (tile_x, tile_y) = tile
                for model_pos in container.model_positions:
                    workload.append((self, container, tile_x, tile_y, model_pos))
            starmap_results = pool.starmap(self.run_workload, workload, chunksize=1)
            pool.close()
            pool.join()

            for tile_x, tile_y, model_pos, tmp_key_byte, tmp_r2s, tmp_betas in starmap_results:
                self.R2s[model_pos] = tmp_r2s
                self.betas[model_pos] = tmp_betas
                tile_index = list(container.tiles).index((tile_x, tile_y))
                self.results[tile_index].append(tmp_key_byte)

    @staticmethod
    def run_workload(self, container, tile_x, tile_y, model_pos):
        num_steps = container.configure(tile_x, tile_y, [model_pos], self.convergence_step)
        if self.convergence_step is None:
            self.convergence_step = np.inf

        model_vals = self.model_value.model.num_vals
        group_n = np.zeros((model_vals), dtype=np.uint32)
        group_mean = np.zeros((model_vals, container.sample_length), dtype=np.float64)
        group_M = np.zeros_like(group_mean)

        r2s = np.empty((num_steps, model_vals, container.sample_length))

        model_input = np.arange(model_vals, dtype='uint8')[..., np.newaxis]
        hypotheses = self.model_value.calculate_table([model_input, None, None, None])

        traces_processed = 0
        converge_index = 0
        for batch in container.get_batches(tile_x, tile_y):
            if traces_processed >= self.convergence_step:
                instant_r2s, _ = self.calculate(group_n, group_mean, group_M, hypotheses)
                r2s[converge_index, :, :] = instant_r2s
                traces_processed = 0
                converge_index += 1
            # Update
            plaintext = batch[0]
            traces = batch[-1]
            group_n, group_mean, group_M = self.update(traces, plaintext, group_n, group_mean, group_M)
            traces_processed += traces.shape[0]

        instant_r2s, betas = self.calculate(group_n, group_mean, group_M, hypotheses)
        r2s[converge_index, :, :] = instant_r2s
        candidate = self.find_candidate(r2s[-1, ...])

        return tile_x, tile_y, model_pos, candidate, r2s, betas
