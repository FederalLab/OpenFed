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


from openfed.common.logging import logger

from ..step import Step, after_download


class AfterDownload(Step):
    step_name = after_download

    def step(self, backend, flag: bool) -> None:
        if flag:  # Download success
            # download is to check others upload version
            if backend.reign.upload_version <= backend.version:
                logger.warning(
                    f"Excepted @{backend.version}, received @{backend.reign.upload_version}, discard.")
                return
            # Fetch data from federated core
            packages  = backend.reign.tensor_indexed_packages
            task_info = backend.reign.task_info

            # Add them to agg
            for agg in backend.agg:
                agg.step(packages, task_info)

            # Add them to reducer
            for reducer in backend.reducer:
                reducer.step(task_info)

            # Increase the total number of received models
            backend.received_numbers += 1
            # Record current reign_task_info, which can be used in the following steps.
            backend.reign_task_info = task_info

            logger.info(
                f"{backend.received_numbers} at v.{backend.version} from {backend.nick_name}.")
        else:
            logger.debug(
                f"Try to download {backend.received_numbers+1} failed.")
