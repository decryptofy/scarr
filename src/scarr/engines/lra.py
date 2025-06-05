# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

from .engine import Engine
from multiprocessing.pool import Pool

import numpy as np
# import numba as nb
import os

import matplotlib.pyplot as plt
# from tqdm import tqdm

# Class to compute average traces
class AverageTraces:
    def __init__(self, num_values, trace_length):
        self.avtraces = np.zeros((num_values, trace_length))
        self.counters = np.zeros(num_values)

    # Method to add a trace and update the average
    def add_trace(self, data, trace):
        if self.counters[data] == 0:
            self.avtraces[data] = trace
        else:
            self.avtraces[data] = self.avtraces[data] + (trace - self.avtraces[data]) / self.counters[data]
        self.counters[data] += 1

    # Method to get data with non-zero counters and corresponding average traces
    def get_data(self):
        avdata_snap = np.flatnonzero(self.counters)
        avtraces_snap = self.avtraces[avdata_snap]
        return avdata_snap, avtraces_snap
    

# Function to compute S-box output
def s_box_out(data, key_byte, sbox):
    s_box_in = data ^ key_byte
    return sbox[s_box_in]


# Linear regression analysis (LRA) for each byte of the key
def lra(data, traces, intermediate_fkt, sbox, sst):
    num_traces, trace_length = traces.shape
    #traces_mean = np.mean(traces, axis=0)
    #sst = np.sum((traces - traces_mean) ** 2, axis=0)
    SSR = np.empty((256, trace_length))
    all_coefs = np.empty((256, 9, trace_length))

    for key_byte in np.arange(256, dtype='uint8'):
        intermediate_var = intermediate_fkt(data, key_byte, sbox)
        M = np.array(list(map(wrapper(8), intermediate_var)))
        # P = np.linalg.pinv(M)
        P = (np.linalg.inv(M.T @ M)) @ M.T

        beta = P @ traces

        E = M @ beta
        
        SSR[key_byte] = np.sum((E - traces) **2 , axis=0)

        all_coefs[key_byte,:] = beta[:]

    R2 = 1 - SSR / sst[None, :]
    return R2, all_coefs


# Function to generate a model of bits
def model_single_bits(x, bit_width):
    model = []
    for i in range(0, bit_width):
        bit = (x >> i) & 1
        model.append(bit)
    model.append(1)
    return model


# Wrapper function to map a model to an intermediate variable
def wrapper(y):
    def curry(x):
        return model_single_bits(x, y)
    return curry


# Function to plot traces
def plot_traces(traces, nb):
    for i in range(nb):
        plt.plot(traces[i])
    plt.show()


def fixed_sst(traces):
    num_traces, trace_length = traces.shape
    u = np.zeros(trace_length)
    v = np.zeros(trace_length)
    for i in range(num_traces):
        u += traces[i]
        v += traces[i]**2
    return np.subtract(v,np.square(u)/num_traces)


class LRA(Engine):
    def __init__(self, key_bytes=np.arange) -> None:
        self.key_bytes = key_bytes
        # initialize values needed
        self.samples_len = 0
        self.traces_len = 0
        self.batch_size = 0
        self.batches_num = 0

        self.average_traces = []
        self.aes_key = []

        self.samples_range = None
        self.samples_start = self.samples_end = 0

        # S-box definition
        self.sbox = np.array([
                    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
                    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
                    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
                    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
                    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
                    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
                    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
                    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
                    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
                    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
                    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
                    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
                    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
                    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
                    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
                    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
                ])


    def update(self, traces: np.ndarray, plaintext: np.ndarray , average_traces):
        for i in range(traces.shape[0]):
            average_traces.add_trace(plaintext[i], traces[i])
        
        return average_traces

        
    def calculate(self, byte_idx, average_traces):
        plain, trace = average_traces.get_data()
        sst = fixed_sst(trace)
        r2, coefs = lra(plain, trace, s_box_out, self.sbox, sst)
        r2_peaks = np.max(r2, axis=1)
        winning_byte = np.argmax(r2_peaks)

        print(f"Key Byte {byte_idx}: {winning_byte:02x}, Max R2: {np.max(r2_peaks):.5f}")
        return winning_byte


    def finalize(self):
        # Show graph?
        pass


    def run(self, container, samples_range=None):
        if samples_range == None:
            self.samples_range = container.data.sample_length
            self.samples_start = 0
            self.samples_end = container.data.sample_length
        else: 
            self.samples_range = samples_range[1]-samples_range[0]
            (self.samples_start, self.samples_end) = samples_range

        # final_results = np.zeros((len(container.tiles), len(container.model_positions), container.sample_lenght))
        self.average_traces = [AverageTraces(256, self.samples_range) for _ in range(len(container.model_positions))]   # all key bytes
        # self.populate(container)
        self.aes_key = [[] for _ in range(len(container.tiles))]

        with Pool(processes=int(os.cpu_count()/2)) as pool:
            workload = []
            for tile in container.tiles:
                (tile_x, tile_y) = tile
                for model_pos in container.model_positions:
                    # self.run_workload(container, tile_x, tile_y, model_pos)
                    workload.append((self, container, self.average_traces[model_pos], tile_x, tile_y, model_pos))
            starmap_results = pool.starmap(self.run_workload, workload, chunksize=1)
            pool.close()
            pool.join()

            for tile_x, tile_y, model_pos, tmp_key_byte in starmap_results:
                tile_index = list(container.tiles).index((tile_x, tile_y))
                self.aes_key[tile_index].append(tmp_key_byte)
        
        # print recovered AES key(s)
        for key in self.aes_key:
            aes_key_bytes = bytes(key)
            print("Recovered AES Key:", aes_key_bytes.hex())


    @staticmethod
    def run_workload(self, container, average_traces, tile_x, tile_y, model_pos):
        container.configure(tile_x, tile_y, [model_pos])

        for batch in container.get_batches(tile_x, tile_y):
            average_traces = self.update(batch[-1][:,self.samples_start:self.samples_end], batch[0], average_traces)

        key_byte = self.calculate(model_pos, average_traces)
        return tile_x, tile_y, model_pos, key_byte