# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import zarr


# Class for handling the iteration and fetching data from the zarr files storing emanation data
class TraceHandler:

    def __init__(self,
                 fileName,
                 batchSize=5000,
                 batchStart=0
                 ) -> None:
        # Path of file containing data
        self.file_name = fileName
        # How many rows of the data to fetch at one time
        self.batch_size = batchSize
        # Current location in the file and save the start index
        self.start = batchStart
        # Establish what the current byte of plaintext/key/ciphertext to fetch
        self.byte = None
        # How many rows are in the current observed directory of the file
        self.data_length = None
        self.slices = None
        # Default directory of the zarr file
        self.current_tile = '0/0'  # Default tile
        # Zarr group
        self.data = zarr.open(fileName, mode='r')
        print("opened zarr file ", fileName)
        self.sample_length = len(self.data['0/0/traces'][0])
        self.traces_length = len(self.data['0/0/traces'])

    # Change the directory to the passed in tile
    def configure(self, tile_x, tile_y, bytes, slice_index=[], trace_index=[], time_slice=[], stride=1, convergence_step = None):
        try:
            self.bytes = bytes
            self.current_tile = f'{tile_x}/{tile_y}'
            self.data_length = len(self.data[f'{self.current_tile}{"/traces"}'])

            if len(slice_index) > 0:
                self.sample_slice = slice_index
            elif len(time_slice) == 2 and stride > 1:
                self.sample_slice = slice(time_slice[0], time_slice[1], stride)
            elif len(time_slice) == 2:
                self.sample_slice = slice(time_slice[0], time_slice[1])
            elif stride > 1:
                self.sample_slice= slice(0, self.sample_length, stride)
            else:
                self.sample_slice = slice(None)
            
            if len(trace_index) > 0:
                self.data_length = len(trace_index)
                self.slices = self.create_batches_index(trace_index)
            else:
                self.slices = self.create_batches()

            if convergence_step is None:
                return 1
            else:
                return -(self.data_length // -convergence_step)

        except:
            print("Error configuring tile")

    # Grab a batch and pass it back to the algorithm's run function
    def grab(self, slice):

        # Declare and select data for the batch
        full_data = [None for i in range(4)]
        for column in self.data[f'{self.current_tile}'].array_keys():
            match column:
                case "plaintext":
                    index = 0
                case "key":
                    index = 1
                case "ciphertext":
                    index = 2
                case _:
                    continue
            full_data[index] = self.select_single_column(column, slice)
        full_data[3] = self.select_traces(slice)
        # Pass batch to the algorithm's run function
        return full_data

    # Fetch data of a non-samples zarr array
    def select_single_column(self, col_name, slice):
        # Grab zarr array
        col_data = self.data[f'{self.current_tile}/{col_name}']
        # Grab the actual data
        data = col_data.get_orthogonal_selection((slice, self.bytes))
        # Pass to grab
        return data
    
    # Specific function for grabbing sample data
    def select_traces(self, slice):
        # Grab zarr array of samples
        traces = self.data[f'{self.current_tile}/traces']
        # Grab the actual data
        data = traces.get_orthogonal_selection((slice, self.sample_slice))
        # Pass to grab
        return data

    def get_batch_generator(self):
        for batch_slice in self.slices:
            yield self.grab(batch_slice)

    def get_batch_index(self, index):
        if index >= len(self.slices):
            return []
        
        return self.grab(self.slices[index])

    def create_batches(self):
        slices = []
        batch_start_index = self.start

        while batch_start_index < self.data_length:
            entry_count = min(self.batch_size, self.data_length - batch_start_index)
            slices.append(slice(batch_start_index, batch_start_index+entry_count))
            batch_start_index += entry_count
        
        return slices
    
    def create_batches_index(self, index):
        slices = []
        batch_start_index = 0

        while batch_start_index < len(index):
            entry_count = min(self.batch_size, len(index) - batch_start_index)
            slices.append(index[batch_start_index:batch_start_index+entry_count])
            batch_start_index += entry_count

        return slices

    def fetch(self, column, tile_x=0, tile_y=0, row_index = 0, col_index=None):
        if col_index is None:
            col_index = slice(None)
        column_data = self.data[f'{tile_x}/{tile_y}/{column}']

        return column_data.get_orthogonal_selection((row_index, col_index))