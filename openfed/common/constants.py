# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from datetime import timedelta

# thread
SLEEP_SHORT_TIME = timedelta(seconds=0.1)
SLEEP_LONG_TIME = timedelta(seconds=5.0)

# communication
DEFAULT_PG_TIMEOUT = timedelta(minutes=30)
DEFAULT_PG_LONG_TIMEOUT = timedelta(minutes=30)
DEFAULT_PG_SHORT_TIMEOUT = timedelta(seconds=1.0)
INTERVAL_AFTER_LAST_FAILED_TIME = 10.0
# if exceed this time, the address will be discarded.
MAX_TRY_TIMES = 5
