"""Service lifecycle management for the control panel.

Each module here exposes a `Service` subclass wrapping a subprocess, docker
container, or in-process task so it can be started, stopped, and queried
uniformly by the ServiceRegistry.
"""
