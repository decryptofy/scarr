# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np

class Filter:
    def __init__(self) -> None:
        pass

    def configure(self, tile_x, tile_y):
        pass

    def filter(self, traces:np.ndarray):
        pass
        