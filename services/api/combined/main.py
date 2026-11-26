"""One application serving Kadan, Hikari and Tsuchi under their own prefixes.

Each service stays a separate FastAPI app with its own routes, settings and
documentation; this module only mounts them and runs their startup and shutdown.
"""

from api.server import app as hikari
from kadan.api.server import app as kadan
from tsuchi.main import app as tsuchi

from combined.mounting import create_app

# Kadan calls Tsuchi and Hikari, so it starts last and stops first.
app = create_app({"tsuchi": tsuchi, "hikari": hikari, "kadan": kadan})
