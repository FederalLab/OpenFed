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


from openfed.common import Array
from openfed.common.base import peeper
from openfed.utils import openfed_class_fmt

# At most case, you are not allowed to modifed this list manually.
peeper.add_to_peeper('countries', dict())


class _Register(Array):

    def __init__(self):
        countries = peeper.get_from_peeper('countries')
        super(_Register, self).__init__(countries)

    @classmethod
    def register_country(cls, country, world):
        countries = peeper.get_from_peeper('countries')
        if country in countries:
            raise KeyError("Already registered.")
        else:
            countries[country] = world

    @classmethod
    def deleted_country(cls, country):
        countries = peeper.get_from_peeper('countries')
        if country in countries:
            if country.is_initialized():
                country.destroy_process_group(
                    group=country.WORLD)

            del countries[country]
            del country

    @classmethod
    def is_registered(cls, country) -> bool:
        countries = peeper.get_from_peeper('countries')
        return country in countries

    @property
    def default_country(self):
        """ If not exists, return None
        """
        return self.default_key

    @property
    def default_world(self):
        """If not exists, return None
        """
        return self.default_value

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Register",
            description=f"Contains {len(self)} countries."
        )


register = _Register()
