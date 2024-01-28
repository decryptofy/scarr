# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
from .engine import Engine
import numba as nb
from multiprocessing.pool import Pool
import asyncio


class MIA(Engine):

    def __init__(self, model, bin_num=9, convergence_step=None) -> None:

        self.model = model
        self.bin_num = bin_num

        self.bins = None
        self.histogram = None

        self.convergence_step = convergence_step
        self.candidates = None
        self.results = None

    def run(self, container):
        final_results = None
        final_candidates = None
        with Pool() as pool:
            workload = []
            for tile in container.tiles:
                (tile_x, tile_y) = tile
                for byte in container.bytes:
                    workload.append((self, container, tile_x, tile_y, byte))

            starmap_results = pool.starmap(self.run_workload, workload)
            pool.close()
            pool.join()

            for tile_x, tile_y, byte, results, candidates in starmap_results:
                if final_results is None:
                    final_results = np.zeros((len(container.tiles),
                                              len(container.bytes),
                                              results.shape[0],
                                              256,
                                              container.sample_length), dtype=np.float64)
                    final_candidates = np.zeros((len(container.tiles),
                                                 len(container.bytes),
                                                 results.shape[0]), dtype=np.uint8)

                byte_index = list(container.bytes).index(byte)
                tile_index = list(container.tiles).index((tile_x, tile_y))
                final_results[tile_index, byte_index] = results
                final_candidates[tile_index, byte_index] = candidates

            self.final_results = final_results
            self.final_candidates = final_candidates

    @staticmethod
    def run_workload(self, container, tile_x, tile_y, byte):

        num_steps = container.configure(tile_x, tile_y, [byte], self.convergence_step)
        if self.convergence_step is None:
            self.convergence_step = np.inf

        self.results = np.empty((num_steps, 256, container.sample_length))
        self.candidates = np.empty((num_steps), dtype=np.uint8)

        if container.fetch_async:
            asyncio.run(self.async_byte_result(container))
        else:
            self.byte_result(byte, tile_x, tile_y, container)

        return tile_x, tile_y, byte, self.results, self.candidates

    def update(self, traces: np.ndarray, plaintext: np.ndarray):
        model = self.model.calculate_table(np.squeeze(plaintext))

        min = self.bins[0]
        max = self.bins[-1]

        bins = self.bin_num

        normx = np.float64(float(bins) / (max - min))

        self.histogram_along_axis(traces, model.astype(np.uint8), bins, min, normx, self.histogram)

    async def async_update(self, traces: np.ndarray, plaintext: np.ndarray):
        model = self.model.calculate_table(np.squeeze(plaintext))

        min = self.bins[0]
        max = self.bins[-1]

        bins = self.bin_num

        normx = np.float64(float(bins) / (max - min))

        self.histogram_along_axis(traces, model.astype(np.uint8), bins, min, normx, self.histogram)

    def calculate(self):
        trace_hist = self.histogram.sum(axis=0)

        trace_counts = trace_hist.sum(axis=1)
        trace_counts[trace_counts == 0] = 1

        trace_pdf = (trace_hist.swapaxes(0, 1) / trace_counts).swapaxes(0, 1)
        trace_pdf[trace_pdf == 0] = 1

        trace_model_counts = self.histogram.sum(axis=2)
        trace_model_counts[trace_model_counts == 0] = 1

        trace_model_pdf = (self.histogram.swapaxes(0, 1).swapaxes(1, 2) / trace_model_counts[:, 0]).swapaxes(1, 2).swapaxes(0, 1)
        trace_model_pdf[trace_model_pdf == 0] = 1

        model_probabilities = trace_model_counts / trace_counts

        trace_entropy = np.sum(trace_pdf * np.log2(trace_pdf), axis=1)
        trace_model_entropy = model_probabilities * np.sum(trace_model_pdf * np.log2(trace_model_pdf), axis=2)

        return trace_model_entropy.sum(axis=0) - trace_entropy

    def get_candidate(self):
        return self.final_candidates

    def find_candidate(self, result):
        return np.unravel_index(np.abs(result).argmax(), result.shape[0:])[0]

    def byte_result(self, byte, tile_x, tile_y, container):

        self.histogram = np.zeros((self.model.num_vals, container.sample_length, self.bin_num, 256), dtype=np.uint16)

        traces_processed = 0
        converge_index = 0
        for batch in container.get_batches(tile_x, tile_y, byte):
            if traces_processed >= self.convergence_step:
                result = self.calculate().swapaxes(0, 1)
                self.results[converge_index, :, :] = result
                self.candidates[converge_index] = self.find_candidate(result)
                traces_processed = 0
                converge_index += 1

            plaintext = batch[0]
            samples = batch[-1]

            if self.bins is None:
                self.bins = np.linspace(np.min(batch[-1]), np.max(batch[-1]), self.bin_num + 1)

            self.update(samples, plaintext)

        result = self.calculate().swapaxes(0, 1)
        self.results[converge_index, :, :] = result
        self.candidates[converge_index] = self.find_candidate(result)

    async def async_byte_result(self, container):
        self.histogram = np.zeros((self.model.num_vals, container.sample_length, self.bin_num, 256), dtype=np.uint16)
        index = 0
        batch = container.get_batch_index(index)
        index += 1

        self.bins = np.linspace(np.min(batch[-1]), np.max(batch[-1]), self.bin_num + 1)

        traces_processed = 0
        converge_index = 0

        while len(batch) > 0:
            if traces_processed >= self.convergence_step:
                result = self.calculate().swapaxes(0, 1)
                self.results[converge_index, :, :] = result
                self.candidates[converge_index] = self.find_candidate(result)
                traces_processed = 0
                converge_index += 1

            task = asyncio.create_task(self.async_update(batch[-1], batch[0]))
            traces_processed += batch[-1].shape[0]
            batch = container.get_batch_index(index)
            index += 1
            await task

        result = self.calculate().swapaxes(0, 1)
        self.results[converge_index, :, :] = result
        self.candidates[converge_index] = self.find_candidate(result)

    @staticmethod
    @nb.njit(parallel=True)
    def histogram_along_axis(data, data2, nx, xmin, normx, count):
        for samples in nb.prange(data.shape[1]):
            local_count = np.empty((count.shape[0], nx, data2.shape[0]), dtype=np.uint16)
            local_count[:, :, :] = 0
            for traces in range(data.shape[0]):
                ix = min(nx-1, (data[traces, samples] - xmin) * normx)
                for key_index in range(data2.shape[0]):
                    local_count[data2[key_index, traces], int(ix), key_index] += nb.uint16(1)

            count[:, samples, :, :] += local_count
