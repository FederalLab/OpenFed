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


from collections import OrderedDict
from typing import Dict

from openfed.common import Array, logger
from openfed.utils import openfed_class_fmt


class Country():
    ...


class World():
    ...


# At most case, you are not allowed to modifed this list manually.
_country: Dict[Country, World] = OrderedDict()


class _Register(Array):

    def __init__(self):
        super(_Register, self).__init__(_country)

    @classmethod
    def register_country(cls, country: Country, world: World):
        if country in _country:
            raise KeyError("Already registered.")
        else:
            _country[country] = world

    @classmethod
    def deleted_country(cls, country: Country):
        if country in _country:
            if country.is_initialized():
                logger.debug(f"Force to delete country: {country}")
                country.destroy_process_group(
                    group=country.WORLD)

            del _country[country]
            del country

    @classmethod
    def is_registered(cls, country: Country) -> bool:
        return country in _country

    @property
    def default_country(self) -> Country:
        """ If not exists, return None
        """
        return self.default_keys

    @property
    def default_world(self) -> World:
        """If not exists, return None
        """
        return self.default_values

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Register",
            description=f"{len(self)} country have been registed."
        )


register = _Register()
