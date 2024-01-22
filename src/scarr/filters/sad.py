# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
import numba as nb
from .filter import Filter

class SAD(Filter):
    def __init__(self, base:np.ndarray, threshold=0, removal=False):
        self.base = base
        self.threshold = threshold

        if(removal):
            self.trace_sync = self.removal_trace_sync
        else:
            self.trace_sync = self.nonremoval_trace_sync

    def configure(self, tile_x, tile_y):
        return super().configure(tile_x, tile_y)
    
    def filter(self, traces: np.ndarray):

        return self.trace_sync(self.base, traces, self.threshold)

    @staticmethod
    @nb.njit(parallel=True)
    def nonremoval_trace_sync(base:np.ndarray, traces:np.ndarray, threshold):
        base_len = len(base)

        for trace in nb.prange(traces.shape[0]):
            curr_trace = traces[trace, :]
            for offset in nb.prange(traces.shape[1] - base_len):
                if np.sum(np.abs(base - curr_trace[offset:base_len+offset])) <= threshold:
                    traces[trace,:] = np.roll(curr_trace, -1*offset)
                    break

        return traces

    @staticmethod
    @nb.njit(parallel=True)
    def removal_trace_sync(base:np.ndarray, traces:np.ndarray, threshold):
        base_len = len(base)
        valid_indices = np.empty((traces.shape[0]), dtype=np.bool_)
        valid_indices[:] = False
        for trace in nb.prange(traces.shape[0]):
            curr_trace = traces[trace, :]
            for offset in nb.prange(traces.shape[1] - base_len):
                if np.sum(np.abs(base - curr_trace[offset:base_len+offset])) <= threshold:
                    traces[trace,:] = np.roll(curr_trace, -1*offset)
                    valid_indices[trace] = True
                    break

        return traces[valid_indices,:]