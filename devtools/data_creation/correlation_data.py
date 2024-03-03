import numpy as np
from src.scarr.model_values.utils import AES_SBOX

class CorrelationData:

    def __init__(self, num_traces, sample_length, model_pos=[0]) -> None:
        self.num_traces = num_traces
        self.sample_length = sample_length
        self.batch_size = 5000
        self.tiles = [(0,0)]
        self.model_positions = model_pos
        self.key = None
        self.plaintext = None
        self.traces = None
        self.fetch_async = True

    def generate_data(self):
        N = self.num_traces # number of traces
        l = self.sample_length # number of points per trace

        self.key = np.random.randint(0, 256, (16)) # just one key = 1x16
        self.plaintext = np.random.randint(0, 256, (N, 16)) # 5000x16 plaintext positions

        self.traces = np.zeros((N, l), dtype=np.int64)

        # Generate random HW traces
        self.traces = np.random.randint(-128, +128, (N, l), dtype=np.int64)

        # Put leakage where it is needed
        for model_pos in range(16):
            leak_plaintext = self.plaintext[:,model_pos]
            leak_sbox_out = AES_SBOX[self.plaintext[:, model_pos] ^ self.key[model_pos]]
            self.traces[:,4+model_pos] = np.subtract(leak_plaintext,128, dtype=np.int16)
            self.traces[:,24+model_pos] = np.subtract(leak_sbox_out,128, dtype=np.int16)

    def configure(self, tile_x, tile_y, model_positions, convergence_step=None):
        self.model_positions = model_positions
        self.slices = []
        batch_start_index = 0
        while batch_start_index < self.num_traces:
            entry_count = min(self.batch_size, self.num_traces - batch_start_index)
            self.slices.append(slice(batch_start_index, batch_start_index+entry_count))
            batch_start_index += entry_count
        
        return 1

    def get_plaintext(self):
        return self.plaintext

    def get_traces(self):
        return self.traces
    
    def get_key(self):
        return self.key
    
    def get_byte_batch(self, slice, model_pos):

        return [self.plaintext[slice, [model_pos]], self.key[[model_pos]], self.traces[slice,:]]
    
    def get_batches_by_byte(self, tile_x, tile_y, model_pos):
        for slice in self.slices:
            yield self.get_byte_batch(slice, model_pos)
    
    def get_batch(self, slice):

        return [self.plaintext[slice,self.model_positions], self.key[self.model_positions], self.traces[slice,:]]
    
    def get_batches_all(self, tile_x, tile_y):
        for slice in self.slices:
            yield self.get_batch(slice)

    def get_batch_index(self, index):

        if index >= len(self.slices):
            return []
        
        return [self.plaintext[self.slices[index], self.model_positions], self.key[self.model_positions], self.traces[self.slices[index], :]]