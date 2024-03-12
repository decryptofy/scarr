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
import mpmath
from functools import lru_cache
import math
np.seterr(divide='ignore', invalid='ignore')

nb.config.THREADING_LAYER = 'workqueue'

# status: experimental
class Chi2Test(Engine):

   def __init__(self, bin_num=9, convergence_step=None, min_thresh=400, processes_num=16) -> None:
       self.bin_num = bin_num

       # All p-value computations of chi2 values beyond this threshold will be approximated
       self.min_thresh = min_thresh
       self.processes_num = processes_num
       self.samples_len = 0
       self.histograms_mins = None
       self.norms = None
       self.convergence_step = convergence_step
       self.num_steps = 0
       self.mask = None
       self.interm_histogram = None

   def populate_histograms(self, container):
       interm_results = np.zeros((len(container.tiles), self.num_steps, 2, self.samples_len, self.bin_num))
       traces = [0,1]
       with Pool() as pool:
           workload = []
           for tile in container.tiles:
               (tile_x, tile_y) = tile
               for trace in traces:
                   workload.append((self, container, tile_x, tile_y, trace))

           starmap_results = pool.starmap(self.run_workload, workload)
           pool.close()
           pool.join()

       for tile_x, tile_y, trace_num, _result in starmap_results:
           tile_index = list(container.tiles).index((tile_x, tile_y))
           interm_results[tile_index, :, trace_num] = _result
       
       self.interm_histogram = interm_results

   def compute_chi2(self, counts1, counts2):
       total_count = counts1 + counts2
       total_sum = np.sum(total_count)
       expected_c1 = (np.sum(counts1) * total_count) / total_sum
       expected_c2 = (np.sum(counts2) * total_count) / total_sum
       diff_c1 = (counts1 - expected_c1)**2
       diff_c2 = (counts2 - expected_c2)**2
       chi2 = np.float64(np.sum(diff_c1 / expected_c1) + np.sum(diff_c2 / expected_c2))
       dof = np.int32(len(counts1) - 1)

       return chi2, dof

   def run(self, container):
       self.samples_len = container.min_samples_length
       with Pool() as pool:
           self.setup_histogram_parameters(container)
           interm_results = np.zeros((len(container.tiles), self.num_steps, 2, self.samples_len, self.bin_num))
           traces = [0,1]
           workload = []
           for tile in container.tiles:
               (tile_x, tile_y) = tile
               for trace in traces:
                   workload.append((self, container, tile_x, tile_y, trace))

           starmap_results = pool.starmap(self.run_workload, workload)

           for tile_x, tile_y, trace_num, _result in starmap_results:
               tile_index = list(container.tiles).index((tile_x, tile_y))
               interm_results[tile_index, :, trace_num] = _result
       
           self.interm_histogram = interm_results

           final_results = np.zeros((len(container.tiles), self.num_steps, self.samples_len), dtype=np.float32)
           calc_workload = [(self, tile_index, sample_index, convergence_step, self.interm_histogram[tile_index, convergence_step, :, sample_index, :]) 
                           for convergence_step in range(self.num_steps) for tile_index in range(len(container.tiles)) for sample_index in range(self.samples_len)]

           starmap_results = pool.starmap(self.run_calculate, calc_workload, chunksize=int(self.samples_len / self.processes_num))
           pool.close()
           pool.join()

           for tile_index, sample_index, convergence_step, p_val in starmap_results:
               final_results[tile_index, convergence_step, sample_index] = p_val

           self.final_results = final_results
   
   @staticmethod
   def run_calculate(self, tile_index, sample_index, convergence_step, frequencies):
       valid_indices1 = frequencies[0] != 0
       valid_indices2 = frequencies[1] != 0
       mask = np.logical_or(valid_indices1, valid_indices2)
       chi2, dof = self.compute_chi2(frequencies[0, mask], frequencies[1, mask])

       # chi2/4 is a good approximation for the min digits of precision required to compute
       mpmath.mp.dps = int(self.min_thresh / 4)
       
       # Rounding and approximations to speed up calculations for high chi2 values
       if chi2 > self.min_thresh:
           mpmath.mp.dps = int(chi2 / 4)

           # Round chi2 to the second leftmost digit for increased cache hits
           d = math.floor(math.log10(chi2)) - 1
           chi2 = round(chi2 / math.pow(10, d)) * math.pow(10, d)

           # Odd dofs take too long to compute, so change to even. 
           if dof % 2: 
               dof = dof + 1
           return tile_index, sample_index, convergence_step, self.calculateRounded(chi2, dof)
       return tile_index, sample_index, convergence_step, self.calculate(chi2, dof)

   # calculates p-value
   def calculate(self, chi2: np.float32, dof: int):
       return -float(mpmath.log10(1-(mpmath.gammainc(dof/2.0, 0, chi2/2.0)/mpmath.gamma(dof/2.0))))

   # Only cache the function calls where chi2 has been rounded
   @lru_cache(maxsize=None)
   def calculateRounded(self, chi2: np.int32, dof: int):
       return -float(mpmath.log10(1-(mpmath.gammainc(dof/2.0, 0, chi2/2.0)/mpmath.gamma(dof/2.0))))
   
   # Determine histogram range for all sample points based on the first batch of traces from both sets
   def setup_histogram_parameters(self, container):
       self.num_steps = container.configure(0, 0, [0], self.convergence_step)
       batch1 = container.get_batch_index(0)
       traces1 = batch1[-1]

       container.configure2(0, 0, [0])
       batch2 = container.get_batch_index2(0)
       traces2 = batch2[-1]

       histograms_maxs = np.maximum(np.max(traces1, axis=0), np.max(traces2, axis=0))
       self.histograms_mins = np.minimum(np.min(traces1, axis=0), np.min(traces2, axis=0))

       self.norms = np.array(float(self.bin_num) / (histograms_maxs - self.histograms_mins), dtype=np.float64)

   @staticmethod
   def run_workload(self, container, tile_x: int, tile_y: int, trace_num: int):
       self.result = np.zeros((self.num_steps, self.samples_len, self.bin_num), dtype=np.uint32)
       self.histogram = np.zeros((self.samples_len, self.bin_num), dtype=np.uint32)

       if(trace_num == 0):
           container.configure(0, 0, [0])
       else:
           container.configure2(0, 0, [0])
   
       if self.convergence_step is None:
           self.convergence_step = np.inf

       if container.fetch_async:
           asyncio.run(self.batch_loop(container, trace_num))
       else:
           traces_processed = 0
           converge_index = 0

           if trace_num == 0:
               fetch = container.get_batches
           else:
               fetch = container.get_batches2
           
           batches = fetch(tile_x, tile_y)
           for batch in batches:
               if traces_processed >= self.convergence_step:
                   self.result[converge_index] = self.histogram
                   traces_processed = 0
                   converge_index += 1

               trace = batch[-1]
               self.internal_state_update(trace, self.bin_num, self.histograms_mins, self.norms, self.histogram)
               traces_processed += trace.shape[0]
           
           self.result[converge_index] = self.histogram
   
       return tile_x, tile_y, trace_num, self.result

   async def batch_loop(self, container, trace_num):
       traces_processed = 0
       converge_index = 0

       if(trace_num == 0):
           fetch = container.get_batch_index
       else:
           fetch = container.get_batch_index2

       index = 0
       while True:
           
           batch = fetch(index)
           if len(batch) <= 0:
               break

           if traces_processed >= self.convergence_step:
               self.result[converge_index] = self.histogram
               traces_processed = 0
               converge_index += 1
           
           traces = batch[-1]
           task = asyncio.create_task(self.async_update(traces))
           traces_processed += traces.shape[0]
           index += 1
           
           await task
       
       self.result[converge_index] = self.histogram
   
   async def async_update(self, traces):
       self.internal_state_update(traces, self.bin_num, self.histograms_mins, self.norms, self.histogram)
   
   @staticmethod
   @nb.njit(parallel=True, nogil=True)
   def internal_state_update(traces: np.ndarray, nx: int, mins: np.ndarray, norms: np.ndarray, count: np.ndarray):
       for sample in nb.prange(traces.shape[1]):
           norm = norms[sample]
           min_val = mins[sample]
           for trace in range(traces.shape[0]):
                   ix = min(nx-1, (traces[trace, sample] - min_val) * norm)
                   count[sample, int(ix)] += nb.uint32(1)
