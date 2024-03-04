# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
from .model_value import ModelValue


class GuessModelValue(ModelValue):

    def calculate_table(self, batch):
        pass

    def calculate_all_tables(self, batch):
        pass

    def calculate_all_tables_helper(self, data):
        pass
