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
    """
    A normalization filter that standardizes trace data using precomputed tile statistics.

    This filter uses the mean and variance for a specific tile, obtained from the Stats object,
    to normalize side-channel trace data to zero mean and unit variance.
    """

    def __init__(self, stats: Stats) -> None:
        """
        Initialize the Normalize filter with global tile statistics.

        Parameters
        ----------
        stats : Stats
            A Stats object that provides tile-level means, variances, and coordinate mappings.
        """
        self.tile_means = stats.get_means()
        self.tile_variances = stats.get_variances()
        self.tiles = stats.get_tiles()

    def configure(self, tile_x, tile_y):
        """
        Select and configure the tile statistics for a specific (x, y) tile.

        Parameters
        ----------
        tile_x : int
            X-coordinate of the tile.
        tile_y : int
            Y-coordinate of the tile.
        """
        tile_index = self.tiles.index((tile_x, tile_y))
        self.mean = self.tile_means[tile_index, :]
        self.variance = self.tile_variances[tile_index, :]

    def filter(self, traces: np.ndarray):
        """
        Apply normalization to a batch of traces using the configured tile statistics.

        Parameters
        ----------
        traces : np.ndarray
            A NumPy array of raw side-channel traces to normalize.

        Returns
        -------
        np.ndarray
            Normalized traces with zero mean and unit variance.
        """
        return (traces - self.mean) / self.variance
