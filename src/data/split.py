from __future__ import annotations

from typing import Iterable


TRAIN_ACTORS = {f"{idx:02d}" for idx in range(1, 17)}
VAL_ACTORS = {f"{idx:02d}" for idx in range(17, 21)}
TEST_ACTORS = {f"{idx:02d}" for idx in range(21, 25)}


def assign_split(actor_id: str) -> str:
    actor_id = str(actor_id).zfill(2)
    if actor_id in TRAIN_ACTORS:
        return "train"
    if actor_id in VAL_ACTORS:
        return "val"
    if actor_id in TEST_ACTORS:
        return "test"
    raise ValueError(f"Unknown actor id: {actor_id}")


def assert_actor_disjointness(actor_ids: Iterable[str]) -> None:
    normalized = {str(actor_id).zfill(2) for actor_id in actor_ids}
    unknown = normalized - TRAIN_ACTORS - VAL_ACTORS - TEST_ACTORS
    if unknown:
        raise ValueError(f"Found actor ids outside official split definition: {sorted(unknown)}")

