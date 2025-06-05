# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

from .engine import Engine
import numpy as np
from multiprocessing.pool import Pool
# import numba as nb
import os


# from tqdm import tqdm
# from typing import List


def model_single_bits(x, bit_width):
    model = []
    for i in range(0, bit_width):
        bit = (x >> i) & 1
        model.append(bit)
    model.append(1)
    return model


def model_mean(byte_idx, model, group_n):
    return np.average(model, axis=0, weights=group_n[:])


def model_cov(byte_idx, model, mod_mean, group_n):
    return np.cov(model - mod_mean[None,:], rowvar=False,
                  fweights=group_n[:])


def covar_with_model(byte_idx, model, mod_mean, grouped_mean, ungrouped_mean, n_traces, group_n):
    # Indexing in two steps is necesssary to get the correct order.
    # https://stackoverflow.com/q/35016092/4071916
    signal_diff = (grouped_mean[:,:] -
                   ungrouped_mean[None,:])
    model_diff = model - mod_mean[None,:]
    return np.tensordot(model_diff * group_n[:,None], signal_diff,
                        axes=(0, 0)) / (n_traces - 1)


def ungrouped_stats(grouped_mean, grouped_var, group_n):
    gm0 = grouped_mean
    gn0 = group_n
    mu  = np.average(gm0, axis=0, weights=gn0)
    var = (np.sum((gm0 - mu[None, :]) ** 2 * gn0[:, None], axis=0) +
           np.sum(grouped_var[0] * (gn0 - 1)[:, None], axis=0))
    var /= max(np.sum(gn0) - 1, 1)
    return mu, var


def lra(byte_idx, model_input_sbox, ungrouped_var, group_n, grouped_mean, ungrouped_mean, n_traces, n_points):
    model_params = np.zeros((256, 16, n_points))
    total_rsq = np.zeros(256)
    
    for key_byte in range(256):
        model_input_ptxt = model_input_sbox[np.arange(256) ^ key_byte]
        mod_mean = model_mean(byte_idx, model_input_ptxt, group_n)
        mod_cov = model_cov(byte_idx, model_input_ptxt, mod_mean, group_n)
        mod_invcov = np.linalg.inv(mod_cov) # (M^T M)^-1 from (1)

        # M^T L from (1)
        cov = covar_with_model(byte_idx, model_input_ptxt, mod_mean, grouped_mean, ungrouped_mean, n_traces, group_n)

        # B from (1)
        model_params[key_byte,:,:] = np.matmul(mod_invcov, cov);

        # Total coefficient of determination, over all sample points.
        total_rsq[key_byte] = (
            np.tensordot(cov / ungrouped_var[None,:],
                         model_params[key_byte,:,:],
                         axes=((0, 1), (0, 1))))

    key_byte = np.argmax(total_rsq)
    real_params = model_params[key_byte,:,:]
    return key_byte, total_rsq, real_params


class LRA(Engine):
    def __init__(self):
        # initialize values needed

        self.key_bytes = np.arange(16)
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

        self.model_input = np.zeros((256, 16), dtype=np.float64)
        # build a bit model table
        for i in range(256):
            self.model_input[i, :8] = [(i >> j) & 1 for j in range(8)]
            self.model_input[i, 8:] = [(self.sbox[i] >> j) & 1 for j in range(8)]

        self.group_n = None
        self.group_mean = None
        self.group_M = None

    # def populate(self, container):
    #     # initialize dimensional variables
    #     self.samples_len = container.sample_length
    #     self.traces_len = container.data.traces_length
    #     self.batch_size = container.data.batch_size
    #     # self.batches_num = (self.traces_len + self.batch_size - 1) // self.batch_size


    # Load traces and plaintext data
    def load_data(self, container, i, model_pos):
        batch = container.get_batch_index(i)
        if not batch:
            return None
        trange = batch[-1][:, 1000:2000].astype(np.float64)
        plaintext = batch[0]
        return trange, plaintext


    # Load data for batches
    def update(self, traces: np.ndarray, plaintext: np.ndarray, group_n, group_mean, group_M):
        num_traces, num_samples = traces.shape

        for i in range(num_traces):
            t = traces[i]
            g = int(plaintext[i])
            n = group_n[g]
            n1 = n + 1
            delta = t - group_mean[g]
            group_mean[g] += delta / n1
            group_M[g] += delta * (t - group_mean[g])
            group_n[g] = n1

        return group_n, group_mean, group_M


    def calculate(self, byte_idx, group_n, group_mean, group_M):
        group_var = np.nan_to_num(
            group_M / np.clip(group_n[:, None] - 1, 1, None))

        un_mu, un_var = ungrouped_stats(group_mean, group_var, group_n)
        n_traces = np.sum(group_n)
        n_points = group_var.shape[1]

        k, r2, _ = lra(byte_idx, self.model_input, un_var,
                        group_n, group_mean, un_mu,
                        n_traces, n_points)
        print("Key Byte {:2d}: {:02x}, Max R2: {:.5f}".format(
            byte_idx, k, np.max(r2)))
        return k


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

        self.aes_key = [[] for _ in range(len(container.tiles))]

        with Pool(processes=int(os.cpu_count()/2)) as pool:
            workload = []
            for tile in container.tiles:
                (tile_x, tile_y) = tile
                for model_pos in container.model_positions:
                    # self.run_workload(container, tile_x, tile_y, model_pos)
                    workload.append((self, container, tile_x, tile_y, model_pos))
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
    def run_workload(self, container, tile_x, tile_y, model_pos):
        container.configure(tile_x, tile_y, model_pos)
        group_n = np.zeros((256), dtype=np.uint32)
        group_mean = np.zeros((256, self.samples_range), dtype=np.float64)
        group_M = np.zeros_like(group_mean)
        # print(">>", group_n.shape, group_mean.shape, group_M.shape)

        for batch in container.get_batches(tile_x, tile_y):
            (group_n, group_mean, group_M) = self.update(batch[-1][:,self.samples_start:self.samples_end], batch[0], group_n, group_mean, group_M)

        key_byte = self.calculate(model_pos, group_n, group_mean, group_M)
        return tile_x, tile_y, model_pos, key_byte