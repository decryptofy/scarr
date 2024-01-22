# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
from .filter import Filter
from ..engines.stats import Stats

class Normalize(Filter):
    def __init__(self, stats:Stats) -> None:
        self.tile_means = stats.get_means()
        self.tile_variances = stats.get_variances()
        self.tiles = stats.get_tiles()

    def configure(self, tile_x, tile_y):
        tile_index = self.tiles.index((tile_x, tile_y))
        self.mean = self.tile_means[tile_index,:]
        self.variance = self.tile_variances[tile_index,:]

    def filter(self, traces:np.ndarray):

        return (traces - self.mean) / self.variance