# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.


from .engine import Engine
from ..model_values.model_value import ModelValue
from ..model_values.model_bits import ModelBits
from multiprocessing.pool import Pool
import numpy as np
import os


# Class to compute average traces
class AverageTraces:
    def __init__(self, num_values, trace_length):
        self.avtraces = np.zeros((num_values, trace_length))
        self.counters = np.zeros(num_values)

    # Method to add a trace and update the average
    def add_trace(self, data, trace):
        if self.counters[data] == 0:
            self.avtraces[data] = trace
        else:
            self.avtraces[data] = self.avtraces[data] + (trace - self.avtraces[data]) / self.counters[data]
        self.counters[data] += 1

    # Method to get data with non-zero counters and corresponding average traces
    def get_data(self):
        avdata_snap = np.flatnonzero(self.counters)
        avtraces_snap = self.avtraces[avdata_snap]
        return avdata_snap, avtraces_snap


class LRA(Engine):
    def __init__(self, model_value: ModelValue, convergence_step=None, normalize=False, bias=True) -> None:
        self.normalize = normalize

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

    def update(self, traces: np.ndarray, plaintext: np.ndarray, average_traces):
        for i in range(traces.shape[0]):
            average_traces.add_trace(plaintext[i], traces[i])
        
        return average_traces
        
    def calculate(self, average_traces, model):
        plain, traces = average_traces.get_data()
        num_traces, trace_length = traces.shape
        model_vals = model.shape[0]
        
        SST = np.sum(np.square(traces), axis=0) - np.square(np.sum(traces, axis=0))/num_traces
        SSR = np.empty((model_vals, trace_length))
        
        # Linear regression analysis (LRA) for each model position
        P = np.linalg.pinv(model)
        betas = P @ traces
        # Below loop is equivalent to:
        #   E =  model @ beta
        #   SSR = np.sum((E - traces)**2, axis=1)
        # However this takes too much memory
        for i in range(0, betas.shape[-1], step := 1):
            E = model @ betas[..., i:i+step]
            SSR[..., i:i+step] = np.sum((E - traces[:, i:i+step])**2, axis=1)
        
        R2s = 1 - SSR / SST[None, :]
        if self.normalize: # Normalization
            R2s = (R2s - np.mean(R2s, axis=0, keepdims=True)) / np.std(R2s, axis=0, keepdims=True)

        return R2s, betas

    def find_candidate(self, R2s):
        r2_peaks = np.max(R2s, axis=1)
        winning_candidate = np.argmax(r2_peaks)
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

            for tile_x, tile_y, model_pos, candidate, r2s, betas in starmap_results:
                self.R2s[model_pos] = r2s
                self.betas[model_pos] = betas
                tile_index = list(container.tiles).index((tile_x, tile_y))
                self.results[tile_index].append(candidate)

    @staticmethod
    def run_workload(self, container, tile_x, tile_y, model_pos):
        num_steps = container.configure(tile_x, tile_y, [model_pos], self.convergence_step)
        if self.convergence_step is None:
            self.convergence_step = np.inf

        model_vals = self.model_value.model.num_vals
        average_traces = AverageTraces(model_vals, container.sample_length)

        r2s = np.empty((num_steps, model_vals, container.sample_length))

        model_input = np.arange(model_vals, dtype='uint8')[..., np.newaxis]
        hypotheses = self.model_value.calculate_table([model_input, None, None, None])

        traces_processed = 0
        converge_index = 0
        for batch in container.get_batches(tile_x, tile_y):
            if traces_processed >= self.convergence_step:
                instant_r2s, _ = self.calculate(average_traces, hypotheses)
                r2s[converge_index, :, :] = instant_r2s
                traces_processed = 0
                converge_index += 1
            # Update
            plaintext = batch[0]
            traces = batch[-1]
            average_traces = self.update(traces, plaintext, average_traces)
            traces_processed += traces.shape[0]

        instant_r2s, betas = self.calculate(average_traces, hypotheses)
        r2s[converge_index, :, :] = instant_r2s
        candidate = self.find_candidate(r2s[-1, ...])

        return tile_x, tile_y, model_pos, candidate, r2s, betas
