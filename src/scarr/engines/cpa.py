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


class CPA(Engine):

    def __init__(self, model, convergence_step=None) -> None:
        self.model = model

        self.trace_count = 0
        self.model_sum = None
        self.model_sq_sum = None

        self.sample_sum = None
        self.sample_sq_sum = None

        self.prod_sum = None

        self.convergence_step = convergence_step
        self.candidate = None
        self.results = None

    def run(self, container):
        self.final_results = None
        self.final_candidates = None
        self.populate(container.sample_length, len(container.bytes))

        with Pool() as pool:
            workload = []
            for tile in container.tiles:
                (tile_x, tile_y) = tile
                workload.append((self, container, tile_x, tile_y))

            starmap_results = pool.starmap(self.run_workload, workload)
            pool.close()
            pool.join()

            for tile_x, tile_y, results, candidates in starmap_results:
                if self.final_results is None:
                    self.final_results = np.zeros((len(container.tiles),
                                                   len(container.bytes),
                                                   results.shape[1],
                                                   256,
                                                   container.sample_length), dtype=np.float64)
                    self.final_candidates = np.zeros((len(container.tiles),
                                                      len(container.bytes),
                                                      results.shape[1]), dtype=np.uint8)

                tile_index = container.tiles.index((tile_x, tile_y))
                self.final_results[tile_index] = results
                self.final_candidates[tile_index] = candidates

    @staticmethod
    def run_workload(self, container, tile_x, tile_y):
        num_steps = container.configure(tile_x, tile_y, container.bytes, self.convergence_step)
        if self.convergence_step is None:
            self.convergence_step = np.inf

        self.results = np.empty((len(container.bytes), num_steps, 256, container.sample_length), dtype=np.float64)
        self.candidates = np.empty((len(container.bytes), num_steps), dtype=np.uint8)

        if container.fetch_async:
            asyncio.run(self.batch_loop(container))

        else:
            traces_processed = 0
            converge_index = 0

            for batch in container.get_batches(tile_x, tile_y):
                if traces_processed >= self.convergence_step:
                    result = self.calculate()
                    self.results[:, converge_index, :, :] = result
                    self.candidates[:, converge_index] = self.find_candidate(result)
                    traces_processed = 0
                    converge_index += 1

                # Generate modeled power values for plaintext values
                model = np.apply_along_axis(self.model.calculate_table, axis=1, arr=batch[0])
                traces = batch[-1].astype(np.float32)

                self.update(traces, model)
                traces_processed += traces.shape[0]

            result = self.calculate()
            self.results[:, converge_index, :, :] = result
            self.candidates[:, converge_index] = self.find_candidate(result)

        return tile_x, tile_y, self.results, self.candidates

    async def batch_loop(self, container):
        index = 0
        batch = container.get_batch_index(index)
        index += 1

        traces_processed = 0
        converge_index = 0

        while len(batch) > 0:
            if traces_processed >= self.convergence_step:
                result = self.calculate()
                self.results[:, converge_index, :, :] = result
                self.candidates[:, converge_index] = self.find_candidate(result)
                traces_processed = 0
                converge_index += 1

            # Generate modeled power values for plaintext values
            model = np.apply_along_axis(self.model.calculate_table, axis=1, arr=batch[0])
            traces = batch[-1].astype(np.float32)

            task = asyncio.create_task(self.async_update(traces, model))
            traces_processed += traces.shape[0]

            batch = container.get_batch_index(index)
            index += 1
            await task

        result = self.calculate()
        self.results[:, converge_index, :, :] = result
        self.candidates[:, converge_index] = self.find_candidate(result)

    async def async_update(self, traces: np.ndarray, data: np.ndarray):
        # Update the number of rows processed
        self.trace_count += traces.shape[0]
        # Update sample accumulator
        self.sample_sum += np.sum(traces, axis=0)
        # Update sample squared accumulator
        self.sample_sq_sum += np.sum(np.square(traces), axis=0)
        # Update model accumulator
        self.model_sum += np.sum(data, axis=0)
        # Update model squared accumulator
        self.model_sq_sum += np.sum(np.square(data), axis=0)
        data = data.reshape((data.shape[0], -1))
        # Update product accumulator
        self.prod_sum += np.matmul(data.T, traces)

    def calculate(self):
        # Sample mean computation
        sample_mean = np.divide(self.sample_sum, self.trace_count)
        # Model mean computation
        model_mean = np.divide(self.model_sum, self.trace_count)

        prod_mean = np.divide(self.prod_sum.reshape(256, self.model_sum.shape[1], self.sample_sum.shape[0]), self.trace_count)
        # Calculate correlation coefficient numerator
        numerator = np.subtract(prod_mean, model_mean[:, :, None]*sample_mean)
        # Calculate correlation coeefficient denominator sample part
        to_sqrt = np.subtract(np.divide(self.sample_sq_sum, self.trace_count), np.square(sample_mean))
        to_sqrt[to_sqrt < 0] = 0
        denom_sample = np.sqrt(to_sqrt)
        # Calculate correlation coefficient denominator model part
        to_sqrt = np.subtract(np.divide(self.model_sq_sum, self.trace_count), np.square(model_mean))
        to_sqrt[to_sqrt < 0] = 0
        denom_model = np.sqrt(to_sqrt)

        denominator = denom_model[:, :, None]*denom_sample

        denominator[denominator == 0] = 1

        return np.divide(numerator, denominator).swapaxes(0, 1)

    def get_candidate(self):
        return self.final_candidates

    def populate(self, sample_length, num_bytes):
        # Sum of the model so far
        self.model_sum = np.zeros((256, num_bytes), dtype=np.float32)
        # Sum of the model squared so far
        self.model_sq_sum = np.zeros((256, num_bytes), dtype=np.float32)
        # Sum of the samples observed
        self.sample_sum = np.zeros((sample_length), dtype=np.float32)
        # Sum of the samples observed squared
        self.sample_sq_sum = np.zeros((sample_length), dtype=np.float32)
        # Sum of the product of the samples and the models
        self.prod_sum = np.zeros((256 * num_bytes, sample_length), dtype=np.float32)

    def update(self, traces: np.ndarray, data: np.ndarray):
        # Update the number of rows processed
        self.trace_count += traces.shape[0]
        # Update sample accumulator
        self.sample_sum += np.sum(traces, axis=0)
        # Update sample squared accumulator
        self.sample_sq_sum += np.sum(np.square(traces), axis=0)
        # Update model accumulator
        self.model_sum += np.sum(data, axis=0)
        # Update model squared accumulator
        self.model_sq_sum += np.sum(np.square(data), axis=0)
        data = data.reshape((data.shape[0], -1))
        # Update product accumulator
        self.prod_sum += np.matmul(data.T, traces)

    def find_candidate(self, result):
        candidate = [None for _ in range(result.shape[0])]

        for i in range(result.shape[0]):
            candidate[i] = np.unravel_index(np.abs(result[i, :, :]).argmax(), result[i, :, :].shape[0:])[0]

        return candidate
