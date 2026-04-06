from db.generation_repository import GenerationInMemoryRepository, GenerationSpacetimeRepository, RunRepository
from db.online_repository import OnlineCapableInMemoryRepository, OnlineCapableSpacetimeRepository, OnlineRunRepository


class InMemoryRepository(OnlineCapableInMemoryRepository):
    pass


class SpacetimeRepository(OnlineCapableSpacetimeRepository):
    pass


__all__ = [
    "InMemoryRepository",
    "OnlineRunRepository",
    "RunRepository",
    "SpacetimeRepository",
    "GenerationInMemoryRepository",
    "GenerationSpacetimeRepository",
]
