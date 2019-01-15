# Copyright (c) 2018-2019 Patricio Cubillos and contributors.
# rate is open-source software under the MIT license (see LICENSE).

from . import VERSION as ver
from .rate import *
from .rate import __all__

__version__ = "{:d}.{:d}.{:d}".format(ver.rate_VER,
                                      ver.rate_MIN, ver.rate_REV)

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)
