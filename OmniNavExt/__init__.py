def import_extensions():
    import OmniNavExt.controllers
    import OmniNavExt.interactions
    import OmniNavExt.metrics
    import OmniNavExt.objects
    import OmniNavExt.robots
    import OmniNavExt.sensors
    import OmniNavExt.tasks


# Note: The extension module is located at OmniNavExt/envset/extension.py
# and is intended for Isaac Sim Omniverse Kit extension loading.
# It's not needed for standalone script execution.
try:
    from .envset.extension import *
except ImportError:
    # Running in standalone mode, extension not required
    pass