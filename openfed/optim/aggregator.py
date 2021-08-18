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

from openfed.common import glue

from .agg_op import AggOp
from .red_op import ReduceOp


class Aggregator(AggOp, ReduceOp):
    """Glue Agg and Reducer into a single class, named Aggregator.

    .. warn::
        Aggregator is an abstract class, it would not be instant.
    """
    def __new__(cls):
        raise RuntimeError("Call `build_container` to create a aggregator.")


def build_aggregator(agg_op: AggOp, reduce_op: ReduceOp = None) -> Aggregator:
    """Build a aggregator to glue agg_op and reducer.

    Args:
        agg: The agg to aggregate tensor.
        reducer: The reducer to reduce task information.
    """
    # An empty reducer is needed.
    reduce_op = ReduceOp() if reduce_op is None else reduce_op
    return glue(agg_op, reduce_op)
