
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import unittest
from scarr.file_handling.trace_handler import TraceHandler


class TestTraceHandler(unittest.TestCase):

    def test_invalid_tile(self):
        handler = TraceHandler.__new__(TraceHandler)
        handler.data = {}

        with self.assertRaises(RuntimeError):
            handler.configure(99, 99, [0])

    def test_valid_tile(self):
        class FakeTraces:
            shape = (3, 5)

        handler = TraceHandler.__new__(TraceHandler)
        handler.data = {"0/0/traces": FakeTraces()}
        handler.create_batches = lambda: []

        self.assertEqual(handler.configure(0, 0, [0]), 1)

