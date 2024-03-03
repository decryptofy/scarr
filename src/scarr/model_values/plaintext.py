# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
from .model_value import ModelValue
from .utils import WEIGHTS, KEYS


class PlainText(ModelValue):

    def __init__(self) -> None:
        self.num_vals = 256
        self.vals = np.arange(256)

    def calculate(self, batch):
        return np.squeeze(batch[0])
