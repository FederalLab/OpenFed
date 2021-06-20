from collections import OrderedDict
from typing import Dict

from openfed.common import DEBUG, Array, logger
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
                if DEBUG.is_debug:
                    logger.info(
                        f"Forece to delete country: {country}")
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

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Reigster",
            description=f"{len(self)} country have been registed."
        )


register = _Register()