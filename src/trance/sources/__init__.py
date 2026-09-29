"""Small, independently imported adapters for local credential sources.

Source modules are deliberately not imported here. Discovery loads each
adapter on demand so optional integrations do not add startup cost.
"""

__all__: list[str] = []
