from starlette.applications import Starlette
from starlette.routing import Mount

from pint_app.apps.viewer_app import app as viewer_app
from pint_app.apps.neighborhood_app import app as neighborhood_app

# Put the specific path first; mount "/" last.
app = Starlette(
    routes=[
        Mount("/neighborhood", app=neighborhood_app),
        Mount("/", app=viewer_app),
    ]
)
