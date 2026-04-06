from __future__ import annotations

from dataclasses import dataclass

import numpy as np


KEY_COUNT = 3
VALUE_LEVELS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)


@dataclass(frozen=True)
class KeyValueMemoryProfile:
    name: str
    active_keys: tuple[int, ...]
    value_levels: tuple[float, ...]
    max_stores: int
    gap_scale: float
    gap_bias: int
    use_separator: bool = True


KV_PROFILES: dict[str, KeyValueMemoryProfile] = {
    "kv_easy": KeyValueMemoryProfile(
        name="kv_easy",
        active_keys=(0, 1),
        value_levels=(0.0, 0.5, 1.0),
        max_stores=2,
        gap_scale=0.4,
        gap_bias=0,
    ),
    "kv_mid": KeyValueMemoryProfile(
        name="kv_mid",
        active_keys=(0, 1, 2),
        value_levels=(0.0, 0.5, 1.0),
        max_stores=2,
        gap_scale=0.75,
        gap_bias=0,
    ),
    "kv_full": KeyValueMemoryProfile(
        name="kv_full",
        active_keys=(0, 1, 2),
        value_levels=VALUE_LEVELS,
        max_stores=3,
        gap_scale=1.0,
        gap_bias=0,
    ),
}


@dataclass(frozen=True)
class KeyValueMemoryTask:
    input_sequences: np.ndarray
    target_sequences: np.ndarray
    step_roles: tuple[tuple[str, ...], ...]
    query_key_ids: np.ndarray
    query_target_ids: np.ndarray
    query_target_values: np.ndarray
    query_store_indices: np.ndarray
    query_distances: np.ndarray
    distractor_loads: np.ndarray
    distractor_counts: np.ndarray
    store_key_ids: np.ndarray
    store_value_ids: np.ndarray
    num_stores: int
    num_queries: int = 1
    input_size: int = 8
    output_size: int = 1
    delay_steps: int = 3
    profile_name: str = "kv_full"
    key_count: int = KEY_COUNT
    value_count: int = len(VALUE_LEVELS)
    value_levels: tuple[float, ...] = VALUE_LEVELS

    @classmethod
    def create(cls, delay_steps: int = 3, *, profile: str = "kv_full") -> "KeyValueMemoryTask":
        profile_spec = _profile_spec(profile)
        num_stores = _num_stores_for_delay(delay_steps, max_stores=profile_spec.max_stores)
        key_orders = _key_orders(profile_spec.active_keys, num_stores)
        value_levels = profile_spec.value_levels
        sequences: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        step_roles: list[tuple[str, ...]] = []
        query_key_ids: list[int] = []
        query_target_ids: list[int] = []
        query_target_values: list[float] = []
        query_store_indices: list[int] = []
        query_distances: list[float] = []
        distractor_loads: list[float] = []
        distractor_counts: list[int] = []
        store_key_ids: list[list[int]] = []
        store_value_ids: list[list[int]] = []

        for order_index, key_order in enumerate(key_orders):
            for pattern_index in range(len(value_levels)):
                value_ids = [
                    (pattern_index + order_index + slot_index) % len(value_levels)
                    for slot_index in range(num_stores)
                ]
                gap_counts = _gap_counts(
                    delay_steps=delay_steps,
                    num_stores=num_stores,
                    offset=order_index + pattern_index,
                    gap_scale=profile_spec.gap_scale,
                    gap_bias=profile_spec.gap_bias,
                )
                query_store_index = (order_index + pattern_index) % num_stores
                query_key = key_order[query_store_index]
                query_target_id = value_ids[query_store_index]
                query_target_value = value_levels[query_target_id]

                sequence_steps: list[list[float]] = []
                target_steps: list[list[float]] = []
                roles: list[str] = []
                query_store_step_index = 0

                for store_index, key_id in enumerate(key_order):
                    if store_index == query_store_index:
                        query_store_step_index = len(sequence_steps)
                    sequence_steps.append(_store_event(key_id, value_levels[value_ids[store_index]]))
                    target_steps.append([0.0])
                    roles.append("store")
                    for distractor_index in range(gap_counts[store_index]):
                        distractor_payload = value_levels[
                            (pattern_index + store_index + distractor_index + 1) % len(value_levels)
                        ]
                        sequence_steps.append(_distractor_event(distractor_payload))
                        target_steps.append([0.0])
                        roles.append("distractor")

                if profile_spec.use_separator:
                    sequence_steps.append(_separator_event())
                    target_steps.append([0.0])
                    roles.append("separator")

                query_step_index = len(sequence_steps)
                sequence_steps.append(_query_event(query_key))
                target_steps.append([query_target_value])
                roles.append("query")

                sequences.append(np.asarray(sequence_steps, dtype=np.float32))
                targets.append(np.asarray(target_steps, dtype=np.float32))
                step_roles.append(tuple(roles))
                query_key_ids.append(int(query_key))
                query_target_ids.append(int(query_target_id))
                query_target_values.append(float(query_target_value))
                query_store_indices.append(int(query_store_index))
                query_distances.append(float(max(0, query_step_index - query_store_step_index - 1)))
                distractor_loads.append(float(sum(gap_counts[query_store_index:])))
                distractor_counts.append(int(sum(gap_counts)))
                store_key_ids.append([int(key_id) for key_id in key_order])
                store_value_ids.append([int(value_id) for value_id in value_ids])

        return cls(
            input_sequences=np.stack(sequences),
            target_sequences=np.stack(targets),
            step_roles=tuple(step_roles),
            query_key_ids=np.asarray(query_key_ids, dtype=np.int32),
            query_target_ids=np.asarray(query_target_ids, dtype=np.int32),
            query_target_values=np.asarray(query_target_values, dtype=np.float32),
            query_store_indices=np.asarray(query_store_indices, dtype=np.int32),
            query_distances=np.asarray(query_distances, dtype=np.float32),
            distractor_loads=np.asarray(distractor_loads, dtype=np.float32),
            distractor_counts=np.asarray(distractor_counts, dtype=np.int32),
            store_key_ids=np.asarray(store_key_ids, dtype=np.int32),
            store_value_ids=np.asarray(store_value_ids, dtype=np.int32),
            num_stores=int(num_stores),
            delay_steps=int(delay_steps),
            profile_name=profile_spec.name,
            key_count=len(profile_spec.active_keys),
            value_count=len(profile_spec.value_levels),
            value_levels=profile_spec.value_levels,
        )


def _profile_spec(profile: str) -> KeyValueMemoryProfile:
    return KV_PROFILES.get(str(profile).strip(), KV_PROFILES["kv_full"])


def _num_stores_for_delay(delay_steps: int, *, max_stores: int) -> int:
    if delay_steps <= 3:
        return min(1, max_stores)
    if delay_steps <= 6:
        return min(2, max_stores)
    return min(3, max_stores)


def _key_orders(active_keys: tuple[int, ...], num_stores: int) -> tuple[tuple[int, ...], ...]:
    if num_stores <= 1:
        return tuple((int(key_id),) for key_id in active_keys)
    orders: list[tuple[int, ...]] = []
    for offset in range(len(active_keys)):
        order = tuple(int(active_keys[(offset + index) % len(active_keys)]) for index in range(num_stores))
        if order not in orders:
            orders.append(order)
    return tuple(orders)


def _gap_counts(
    *,
    delay_steps: int,
    num_stores: int,
    offset: int,
    gap_scale: float,
    gap_bias: int,
) -> tuple[int, ...]:
    effective_delay = max(1, int(round(max(1, int(delay_steps)) * float(gap_scale))) + int(gap_bias))
    base = effective_delay // max(1, num_stores)
    remainder = effective_delay % max(1, num_stores)
    counts = [
        base + (1 if ((offset + gap_index) % num_stores) < remainder else 0)
        for gap_index in range(num_stores)
    ]
    return tuple(max(0, int(count)) for count in counts)


def _store_event(key_id: int, value: float) -> list[float]:
    return [1.0, 0.0, 0.0, 0.0, *_key_one_hot(key_id), float(value)]


def _distractor_event(value: float) -> list[float]:
    return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, float(value)]


def _separator_event() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def _query_event(key_id: int) -> list[float]:
    return [0.0, 1.0, 0.0, 0.0, *_key_one_hot(key_id), 0.0]


def _key_one_hot(key_id: int) -> tuple[float, float, float]:
    return tuple(1.0 if index == int(key_id) else 0.0 for index in range(KEY_COUNT))
