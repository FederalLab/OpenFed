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

from ..hooks import Hooks

after_destroy = 'AFTER_DESTROY'
after_download = 'AFTER_DOWNLOAD'
after_upload = 'AFTER_UPLOAD'

at_first = "AT_FIRST"
at_failed = 'AT_FAILED'
at_invalid_state = 'AT_INVALID_STATE'
at_last = 'AT_LAST'
at_new_episode = 'AT_NEW_EPISODE'
at_zombie = 'AT_ZOMBIE'

before_destroy = 'BEFORE_DESTROY'
before_download = 'BEFORE_DOWNLOAD'
before_upload = 'BEFORE_UPLOAD'


class Step(Hooks):
    """Step hook used for openfed_api.
    """

    def before_destroy(self, leader, *args, **kwargs) -> bool:
        return True

    def before_download(self, leader, *args, **kwargs) -> bool:
        return True

    def before_upload(self, leader, *args, **kwargs) -> bool:
        return True

    def after_destroy(self, leader, *args, **kwargs):
        ...

    def after_download(self, leader, *args, **kwargs):
        ...

    def after_upload(self, leader, *args, **kwargs):
        ...

    def at_failed(self, leader, *args, **kwargs):
        ...

    def at_invalid_state(self, leader, *args, **kwargs):
        ...

    def at_last(self, leader, *args, **kwargs):
        ...

    def at_first(self, leader, *args, **kwargs):
        ...

    def at_new_episode(self, leader, *args, **kwargs):
        ...

    def at_zombie(self, leader, *args, **kwargs):
        ...

    def __call__(self, leader, step_name, *args, **kwargs):
        func = getattr(self, step_name.lower())
        return func(leader, *args, **kwargs)
