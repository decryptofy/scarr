# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
from .engine import Engine
from multiprocessing.pool import Pool
import asyncio


class Stats(Engine):
    def __init__(self):
        self.means = None
        self.variances = None

    def run(self, container):
        tile_means = np.zeros((len(container.tiles), container.sample_length), dtype=np.float64)
        tile_variances = np.zeros((len(container.tiles), container.sample_length), dtype=np.float64)

        with Pool() as pool:
            workload = []

            self.tiles = list(container.tiles)

            for tile in self.tiles:
                (tile_x, tile_y) = tile

                workload.append((self, container, tile_x, tile_y))

            starmap_results = pool.starmap(self._run, workload)

            for tile_x, tile_y, mean, variance in starmap_results:
                tile_index = self.tiles.index((tile_x, tile_y))
                tile_means[tile_index, :] = mean
                tile_variances[tile_index, :] = variance

            self.means = tile_means
            self.variances = tile_variances

    @staticmethod
    def _run(self, container, tile_x, tile_y):
        self.count = np.uint32(0)
        self.mean = np.zeros((container.sample_length), dtype=np.float64)
        self.variance = np.zeros((container.sample_length), dtype=np.float64)

        container.configure(tile_x, tile_y, [0])
        if container.fetch_async:
            asyncio.run(self.stat_batch_loop(container))
        else:
            for batch in container.get_batches(tile_x, tile_y, 0):
                self.update(batch[-1])

        return tile_x, tile_y, self.mean, self.variance / self.count

    async def stat_batch_loop(self, container):
        index = 0
        batch = container.get_batch_index(index)
        index += 1

        while len(batch) > 0:
            task = asyncio.create_task(self.async_update(batch[-1]))
            batch = container.get_batch_index(index)
            index += 1
            await task

    def update(self, traces: np.ndarray):
        self.count += traces.shape[0]

        delta1 = traces - self.mean

        self.mean += np.sum(delta1 / self.count, axis=0)

        delta2 = traces - self.mean

        self.variance += np.sum(delta1 * delta2, axis=0)

    async def async_update(self, traces: np.ndarray):
        self.count += traces.shape[0]

        delta1 = traces - self.mean

        self.mean += np.sum(delta1 / self.count, axis=0)

        delta2 = traces - self.mean

        self.variance += np.sum(delta1 * delta2, axis=0)

    def get_means(self):
        return self.means

    def get_variances(self):
        return self.variances

    def get_tiles(self):
        return self.tiles
