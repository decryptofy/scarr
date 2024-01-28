# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
from multiprocessing.pool import Pool
import os
import asyncio


class Engine:
    """
    Base class that engines inherit from.
    """
    def __init__(self):
        pass

    def run(self, container):
        final_results = np.zeros((len(container.tiles), len(container.bytes), container.sample_length), dtype=np.float32)
        # with Pool(processes=int(os.cpu_count()/2),maxtasksperchild=1000) as pool: #used for benchmarking
        with Pool(processes=int(os.cpu_count()/2)) as pool:
            workload = []
            for tile in container.tiles:
                (tile_x, tile_y) = tile
                for byte in container.bytes:
                    workload.append((self, container, tile_x, tile_y, byte))
            starmap_results = pool.starmap(self.run_workload, workload, chunksize=1)  # Possibly more testing needed
            pool.close()
            pool.join()

            for tile_x, tile_y, byte_pos, tmp_result in starmap_results:
                tile_index = list(container.tiles).index((tile_x, tile_y))
                byte_index = list(container.bytes).index(byte_pos)
                final_results[tile_index, byte_index] = tmp_result

            self.final_results = final_results

    @staticmethod
    def run_workload(self, container, tile_x, tile_y, byte):
        self.populate(container.sample_length)
        container.configure(tile_x, tile_y, [byte])
        if container.fetch_async:
            asyncio.run(self.batch_loop(container))
        else:
            for batch in container.get_batches(tile_x, tile_y, byte):
                self.update(batch[-1], np.squeeze(batch[0]))

        return tile_x, tile_y, byte, self.calculate()

    async def batch_loop(self, container):
        index = 0
        batch = container.get_batch_index(index)
        index += 1

        while len(batch) > 0:
            task = asyncio.create_task(self.async_update(batch[-1], np.squeeze(batch[0])))
            batch = container.get_batch_index(index)
            index += 1
            await task

    def update(self, traces: np.ndarray, plaintext: np.ndarray):
        """
        Function that updates the statistics of the algorithm to be called by the container class.
        Gets passed in an array of traces and an array of plaintext from the trace_handler class.
        Returns None.
        """
        pass

    async def async_update(self, traces: np.ndarray, plaintext: np.ndarray):
        """
        Function that updates the statistics of the algorithm to be called by the container class.
        Gets passed in an array of traces and an array of plaintext from the trace_handler class.
        Returns None.
        """
        pass

    def calculate(self):
        pass

    def get_result(self):
        return self.final_results

    def populate(self, sample_length):
        """
        Function to initialize the member of objects of a given algorithm once the trace_length is known.
        Gets passed in at least sample_length which is the length of one sample trace.
        Returns None.
        """
        pass

    def get_points(self, lower_lim, tile_index=0, byte_index=0,):
        return list(np.where(np.abs(self.final_results[tile_index, byte_index]) >= lower_lim)[0])
