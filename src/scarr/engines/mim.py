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


class MIM(Engine):
    def __init__(self, bin_num=9):

        self.bin_num = bin_num
        self.bins = None

        self.final_results = None
        self.histogram = None

    def run(self, container):
        histogram = np.zeros((len(container.tiles),
                              len(container.bytes),
                              256,
                              container.sample_length,
                              self.bin_num), dtype=np.uint16)
        with Pool() as pool:
            tile_workload = []
            hist_workload = []
            for tile in container.tiles:
                (tile_x, tile_y) = tile
                tile_workload.append((self, container, tile_x, tile_y))
                for byte in container.bytes:
                    hist_workload.append((self, container, tile_x, tile_y, byte))

            starmap_results = pool.starmap(self.run_workload, hist_workload)
            pool.close()
            pool.join()

            for tile_x, tile_y, byte_pos,  result in starmap_results:
                tile_index = list(container.tiles).index((tile_x, tile_y))
                byte_index = list(container.bytes).index(byte_pos)
                histogram[tile_index, byte_index] = result

            self.histogram = histogram
            self.final_results = self.calculate()

    @staticmethod
    def run_workload(self, container, tile_x, tile_y, byte):
        self.byte_histogram = np.zeros((256, container.sample_length, self.bin_num), dtype=np.uint16)
        container.configure(tile_x, tile_y, [byte])
        if container.fetch_async:
            asyncio.run(self.batch_loop(container))
        else:
            for batch in container.get_batches(tile_x, tile_y, byte):
                if self.bins is None:
                    min = np.min(batch[-1])
                    max = np.max(batch[-1])
                    self.bins = np.linspace(min, max, self.bin_num + 1)
                    self.norm = np.float64(float(self.bin_num) / (max - min))

                data = np.bitwise_xor(np.squeeze(batch[0]), np.squeeze(batch[1]), dtype=np.uint8)
                self.histogram_along_axis(batch[-1], data, self.bin_num, self.bins[0], self.norm, self.byte_histogram)

        return tile_x, tile_y, byte, self.byte_histogram

    async def batch_loop(self, container):
        index = 0
        batch = container.get_batch_index(index)
        index += 1

        min = float(np.min(batch[-1]))
        max = float(np.max(batch[-1]))

        self.bins = np.linspace(min, max, self.bin_num + 1)

        self.norm = np.float64(float(self.bin_num) / (max - min))

        while len(batch) > 0:
            data = np.bitwise_xor(np.squeeze(batch[0]), np.squeeze(batch[1]), dtype=np.uint8)
            task = asyncio.create_task(self.async_update(batch[-1], data))
            batch = container.get_batch_index(index)
            index += 1
            await task

    async def async_update(self, traces: np.ndarray, data: np.ndarray):

        self.histogram_along_axis(traces=traces,
                                  data=data, nx=self.bin_num,
                                  xmin=self.bins[0],
                                  normx=self.norm,
                                  count=self.byte_histogram)

    @staticmethod
    @nb.njit(parallel=True)
    def histogram_along_axis(traces, data, nx, xmin, normx, count):
        for sample in nb.prange(traces.shape[1]):
            local_count = np.empty((256, nx), dtype=np.uint16)
            local_count[:, :] = 0
            for trace in range(traces.shape[0]):
                ix = min(nx-1, (traces[trace, sample] - xmin) * normx)
                local_count[data[trace], int(ix)] += 1

            count[:, sample, :] += local_count

    def calculate(self):
        trace_hist = self.histogram.sum(axis=2)

        trace_counts = trace_hist.sum(axis=3)
        trace_counts[trace_counts == 0] = 1

        # trace_pdf = (trace_hist.swapaxes(2,3) / trace_counts).swapaxes(3, 2)
        trace_pdf = (trace_hist.swapaxes(2, 3) / trace_counts[:, :, None, :]).swapaxes(3, 2)
        trace_pdf[trace_pdf == 0] = 1

        trace_profile_counts = self.histogram.sum(axis=4)
        trace_profile_counts[trace_profile_counts == 0] = 1

        trace_profile_pdf = (self.histogram.swapaxes(3, 4).swapaxes(2, 3) / trace_profile_counts[:, 0]).swapaxes(2, 3).swapaxes(3, 4)
        trace_profile_pdf[trace_profile_pdf == 0] = 1

        # profile_probabilities = trace_profile_counts / trace_counts
        profile_probabilities = trace_profile_counts / trace_counts[:, :, None, :]

        trace_entropy = np.sum(trace_pdf * np.log2(trace_pdf), axis=3)

        trace_profile_entropy = profile_probabilities * np.sum(trace_profile_pdf * np.log2(trace_profile_pdf), axis=4)

        del self.histogram

        return trace_profile_entropy.sum(axis=2) - trace_entropy
