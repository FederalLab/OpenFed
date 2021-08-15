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

after_destroy = 'after_destroy'
after_download = 'after_download'
after_upload = 'after_upload'

at_first = "at_first"
at_failed = 'at_failed'
at_invalid_state = 'at_invalid_state'
at_last = 'at_last'
at_new_episode = 'at_new_episode'
at_zombie = 'at_zombie'

before_destroy = 'before_destroy'
before_download = 'before_download'
before_upload = 'before_upload'


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
