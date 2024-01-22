# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np

class Model:

    def __init__(self) -> None:
        pass

    def calculate(self, plaintext: np.ndarray, key):
        pass

    def calculate_table(self, plaintext: np.ndarray):
        pass

    def calculate_profile(self, batch):
        pass
