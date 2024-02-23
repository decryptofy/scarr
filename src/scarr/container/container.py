# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

from dataclasses import dataclass
from typing import Optional

from ..engines.engine import Engine
from ..file_handling.trace_handler import TraceHandler

import math

@dataclass
class ContainerOptions:
    engine: Engine
    handler: TraceHandler
    handler2: Optional[TraceHandler] = None

class Container:
    
    """
    Constructor Parameters:
    - options(see above): Mandatory pass ins see below for individual definitions
        - engine: Engine object that is running computations on the passed in data
        - handler: TraceHandler object that fetches the batches of data to be passed to the algorithm
    - byte_positions: The byte positions to be processed by the algorithm has a default value of [0]
    - tile_positions: The tile positions that are to have the byte positions processed default value of [(0,0)]
    """
    def __init__(self, options: ContainerOptions, Async = True, byte_positions = [0], tile_positions = [(0,0)], filters = [], points=[], trace_index=[], slice=[], stride=1) -> None:
        self.engine = options.engine
        self.data = options.handler
        self.data2 = options.handler2  # second trace (only t-test)

        self.fetch_async = Async

        self.bytes = byte_positions
        self.tiles = tile_positions

        self.filters = filters

        self.trace_index = trace_index

        #t-test data only
        self.min_samples_length = 0
        self.min_traces_length = 0

        if len(points) > 0:
            self.slice_index = points
            self.time_slice = []
            self.stride = 1
            self.sample_length = len(points)
        elif len(slice) == 2 and stride > 1:
            self.slice_index = []
            self.time_slice = slice
            self.stride = stride
            self.sample_length = math.ceil((self.time_slice[1] - self.time_slice[0]) / stride)
        elif len(slice) == 2:
            self.slice_index = []
            self.time_slice = slice
            self.stride = 1
            self.sample_length = int(self.time_slice[1] - self.time_slice[0])
        elif stride > 1:
            self.slice_index = []
            self.time_slice = []
            self.stride = stride
            self.sample_length = math.ceil(self.data.sample_length / self.stride)
        else:
            self.slice_index = []
            self.time_slice = []
            self.stride = 1
            self.sample_length = self.data.sample_length
        
        # second trace additions (only t-test)
        if self.data2 is not None:
            self.sample_length = min(self.sample_length, math.ceil(self.data2.sample_length / self.stride))
            self.min_samples_length = self.sample_length
            self.min_traces_length = min(self.data.traces_length, self.data2.traces_length)

    def run(self):
        self.engine.run(self)

    def configure(self, tile_x, tile_y, bytes, convergence_step = None):
        for filter in self.filters:
            filter.configure(tile_x, tile_y)
        # int() casting needed for random typing linux bug
        return int(self.data.configure(tile_x, tile_y, bytes, self.slice_index, self.trace_index, self.time_slice, self.stride, convergence_step))

    def configure2(self, tile_x, tile_y, bytes, convergence_step = None):
        for filter in self.filters:
            filter.configure(tile_x, tile_y)
        # int() casting needed for random typing linux bug
        return int(self.data2.configure(tile_x, tile_y, bytes, self.slice_index, self.trace_index, self.time_slice, self.stride, convergence_step))

    def get_batches(self, tile_x, tile_y):
        for batch in self.data.get_batch_generator():
            for filter in self.filters:
                batch[-1] = filter.filter(batch[-1])
            yield batch
    
    def get_batches2(self, tile_x, tile_y):
        for batch in self.data2.get_batch_generator():
            for filter in self.filters:
                batch[-1] = filter.filter(batch[-1])
            yield batch
    
    def get_batch_index(self, index):
        batch = self.data.get_batch_index(index)
        if len(batch) > 0:
            for filter in self.filters:
                batch[-1] = filter.filter(batch[-1])
        return batch
    
    def get_batch_index2(self, index):
        batch = self.data2.get_batch_index(index)
        if len(batch) > 0:
            for filter in self.filters:
                batch[-1] = filter.filter(batch[-1])
        return batch
    
    def get_result(self):
        return self.engine.get_result()
