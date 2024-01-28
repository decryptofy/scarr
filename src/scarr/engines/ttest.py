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
np.seterr(divide='ignore', invalid='ignore')

nb.config.THREADING_LAYER = 'workqueue'


class Ttest(Engine):

    def __init__(self) -> None:
        self.sums1 = None
        self.sums2 = None
        self.sums_sq1 = None
        self.sums_sq2 = None
        self.traces_len = 0
        self.samples_len = 0
        self.batch_size = 0
        # Results
        self.final_results = None

    def populate(self, trace):
        if trace == 0:
            self.sums1 = np.zeros((self.samples_len), dtype=np.float32)
            self.sums_sq1 = np.zeros((self.samples_len), dtype=np.float32)
        else:
            self.sums2 = np.zeros((self.samples_len), dtype=np.float32)
            self.sums_sq2 = np.zeros((self.samples_len), dtype=np.float32)

    def accumulate_batch(self, batches, sum, sum_sq):
        # Run accumulation algorithm on each batch
        for batch in batches:
            self.internal_state_update(batch[-1], sum, sum_sq)

    @staticmethod
    @nb.njit(parallel=True, nogil=True)
    def internal_state_update(trace: np.ndarray, sum, sum_sq):
        for sample in nb.prange(trace.shape[1]):
            # Allocate new array for this index's column
            trace_column = np.empty(trace.shape[0], dtype=np.float32)
            trace_column[:] = trace[:, sample]
            # Accumulate sums and sums squared values
            sum[sample] += trace_column.sum()
            sum_sq[sample] += trace_column.T @ trace_column

    async def async_update(self, trace: np.ndarray, trace_number):
        # Run one-pass algorithm on correct traces
        if trace_number == 0:
            self.internal_state_update(trace, self.sums1, self.sums_sq1)
        else:
            self.internal_state_update(trace, self.sums2, self.sums_sq2)

    def calculate(self, trace):
        results = np.zeros((2, self.samples_len), dtype=np.float32)
        # Calculate samples' first order statistics
        if trace == 0:
            results[0] = np.divide(self.sums1, self.traces_len)
            results[1] = np.subtract(np.divide(self.sums_sq1, self.traces_len), (results[0]**2))
        else:
            results[0] = np.divide(self.sums2, self.traces_len)
            results[1] = np.subtract(np.divide(self.sums_sq2, self.traces_len), (results[0]**2))

        return results

    def run(self, container):
        # Initialize dimensional variables and populate arrays
        self.samples_len = container.min_samples_length
        self.traces_len = container.min_traces_length
        self.batch_size = container.data.batch_size
        interm_results = np.empty((len(container.tiles), 2, 2, container.min_samples_length), dtype=np.float32)
        final_results = np.zeros((len(container.tiles), container.min_samples_length), dtype=np.float32)
        trace_counts = [0, 1]
        # Begin multiprocess pool of tasks
        pool = Pool()
        workload = []
        for tile in container.tiles:
            (tile_x, tile_y) = tile
            for trace in trace_counts:
                workload.append((self, container, tile_x, tile_y, trace))
        starmap_results = pool.starmap(self.run_workload, workload)
        pool.close()
        pool.join()

        # Get pool map results
        for tile_x, tile_y, _result, trace in starmap_results:
            tile_index = list(container.tiles).index((tile_x, tile_y))
            interm_results[tile_index, trace] = _result
        # Compute each tile's t-values
        for tile in range(len(container.tiles)):
            final_results[tile] = np.divide(interm_results[tile, 0, 0] - interm_results[tile, 1, 0],
                                            np.sqrt(np.divide(interm_results[tile, 0, 1] + interm_results[tile, 1, 1],
                                            self.traces_len)))

        self.final_results = final_results

    @staticmethod
    def run_workload(self, container, tile_x, tile_y, trace):
        self.populate(trace)
        if container.fetch_async:
            if trace == 0:
                container.configure(tile_x, tile_y, [0])
            else:
                container.configure2(tile_x, tile_y, [0])
            asyncio.run(self.batch_loop(container, trace))
        else:
            if trace == 0:
                # Run accumulation algorithm on the first traces
                container.configure(tile_x, tile_y, [0])
                batches = container.get_batches(tile_x, tile_y)
                self.accumulate_batch(batches, self.sums1, self.sums_sq1)
            elif trace == 1:
                # Run accumulation algorithm on the second traces
                container.configure2(tile_x, tile_y, [0])
                batches = container.get_batches2(tile_x, tile_y)
                self.accumulate_batch(batches, self.sums2, self.sums_sq2)
        # Done
        return tile_x, tile_y, self.calculate(trace), trace

    async def batch_loop(self, container, trace_number):
        index = 0
        while True:
            batch = container.get_batch_index(index) if trace_number == 0 else container.get_batch_index2(index)
            if len(batch) <= 0:
                break
            trace = batch[-1]
            task = asyncio.create_task(self.async_update(trace, trace_number))
            index += 1
            await task
