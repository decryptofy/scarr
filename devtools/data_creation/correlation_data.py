import numpy as np
from src.scarr.models.utils import AES_SBOX

class CorrelationData:

    def __init__(self, num_traces, sample_length, bytes=[0]) -> None:
        self.num_traces = num_traces
        self.sample_length = sample_length
        self.batch_size = 5000
        self.tiles = [(0,0)]
        self.bytes = bytes
        self.key = None
        self.plaintext = None
        self.traces = None
        self.fetch_async = True

    def generate_data(self):
        N = self.num_traces # number of traces
        l = self.sample_length # number of points per trace

        self.key = np.random.randint(0, 256, (16)) # just one key = 1x16
        self.plaintext = np.random.randint(0, 256, (N, 16)) # 5000x16 plaintext bytes

        self.traces = np.zeros((N, l), dtype=np.int64)

        # Generate random HW traces
        self.traces = np.random.randint(-128, +128, (N, l), dtype=np.int64)

        # Put leakage where it is needed
        for byte in range(16):
            leak_plaintext = self.plaintext[:,byte]
            leak_sbox_out = AES_SBOX[self.plaintext[:, byte] ^ self.key[byte]]
            self.traces[:,4+byte] = np.subtract(leak_plaintext,128, dtype=np.int16)
            self.traces[:,24+byte] = np.subtract(leak_sbox_out,128, dtype=np.int16)

    def configure(self, tile_x, tile_y, bytes, convergence_step=None):
        self.bytes = bytes
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
    
    def get_byte_batch(self, slice, byte):

        return [self.plaintext[slice, [byte]], self.key[[byte]], self.traces[slice,:]]
    
    def get_batches_by_byte(self, tile_x, tile_y, byte):
        for slice in self.slices:
            yield self.get_byte_batch(slice, byte)
    
    def get_batch(self, slice):

        return [self.plaintext[slice,self.bytes], self.key[self.bytes], self.traces[slice,:]]
    
    def get_batches_all(self, tile_x, tile_y):
        for slice in self.slices:
            yield self.get_batch(slice)

    def get_batch_index(self, index):

        if index >= len(self.slices):
            return []
        
        return [self.plaintext[self.slices[index], self.bytes], self.key[self.bytes], self.traces[self.slices[index], :]]