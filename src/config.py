from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class TaskConfig:
    name: str = "xor"
    activation_steps: int = 5
    temporal_delay_steps: int = 1
    evaluation_delay_steps: tuple[int, ...] = ()
    key_value_profile: str = "kv_full"
    curriculum_enabled: bool = False
    curriculum_phase_switch_generation: int = 6
    curriculum_phase_split_generation: int | None = None
    curriculum_phase_1_delay_steps: tuple[int, ...] = ()
    curriculum_phase_2_delay_steps: tuple[int, ...] = ()
    curriculum_phase_1_key_value_profile: str = ""
    curriculum_phase_2_key_value_profile: str = ""


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    mode: str = "generation"
    generations: int = 15
    elite_top_k: int = 3
    run_id_prefix: str = "xor-v1"
    variant: str = "stateful"


@dataclass(frozen=True)
class OnlineConfig:
    active_population_size: int = 24
    max_steps: int = 120
    replacement_interval: int = 8
    metrics_interval: int = 4
    success_window: int = 20
    rolling_score_alpha: float = 0.35
    hall_of_fame_top_k: int = 16
    worker_id: str = "local-worker"


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 48
    species_size: int = 8
    max_nodes: int = 24
    max_conns: int = 48
    survival_threshold: float = 0.2
    compatibility_threshold: float = 2.0
    species_elitism: int = 1
    genome_elitism: int = 2
    max_stagnation: int = 12
    spawn_number_change_rate: float = 0.5
    min_species_size: int = 2
    species_number_calculate_by: str = "rank"


@dataclass(frozen=True)
class MutationConfig:
    conn_add: float = 0.25
    conn_delete: float = 0.0
    node_add: float = 0.12
    node_delete: float = 0.0
    weight_mutate_power: float = 0.35
    weight_mutate_rate: float = 0.8
    weight_replace_rate: float = 0.05
    weight_init_std: float = 1.0
    weight_lower_bound: float = -4.0
    weight_upper_bound: float = 4.0
    enabled_toggle_rate: float = 0.08
    eta_mutate_power: float = 0.05
    eta_mutate_rate: float = 0.3
    eta_replace_rate: float = 0.02
    eta_init_mean: float = 0.0
    eta_init_std: float = 0.05
    eta_lower_bound: float = -0.5
    eta_upper_bound: float = 0.5
    plastic_a_mutate_power: float = 0.1
    plastic_a_mutate_rate: float = 0.3
    plastic_a_replace_rate: float = 0.02
    plastic_a_init_mean: float = 1.0
    plastic_a_init_std: float = 0.25
    plastic_a_lower_bound: float = -2.0
    plastic_a_upper_bound: float = 2.0
    plastic_d_mutate_power: float = 0.05
    plastic_d_mutate_rate: float = 0.3
    plastic_d_replace_rate: float = 0.02
    plastic_d_init_mean: float = 0.0
    plastic_d_init_std: float = 0.05
    plastic_d_lower_bound: float = -1.0
    plastic_d_upper_bound: float = 0.0
    bias_mutate_power: float = 0.25
    bias_mutate_rate: float = 0.7
    bias_replace_rate: float = 0.05
    bias_init_std: float = 0.75
    bias_lower_bound: float = -3.0
    bias_upper_bound: float = 3.0
    alpha_mutate_power: float = 0.15
    alpha_mutate_rate: float = 0.6
    alpha_replace_rate: float = 0.05
    alpha_init_mean: float = 0.35
    alpha_init_std: float = 0.15
    alpha_slow_mutate_power: float = 0.08
    alpha_slow_mutate_rate: float = 0.6
    alpha_slow_replace_rate: float = 0.05
    alpha_slow_init_mean: float = 0.85
    alpha_slow_init_std: float = 0.08
    slow_input_gain_mutate_power: float = 0.1
    slow_input_gain_mutate_rate: float = 0.6
    slow_input_gain_replace_rate: float = 0.05
    slow_input_gain_init_mean: float = 0.35
    slow_input_gain_init_std: float = 0.12
    slow_input_gain_lower_bound: float = -2.0
    slow_input_gain_upper_bound: float = 2.0
    slow_output_gain_mutate_power: float = 0.15
    slow_output_gain_mutate_rate: float = 0.6
    slow_output_gain_replace_rate: float = 0.05
    slow_output_gain_init_mean: float = 1.0
    slow_output_gain_init_std: float = 0.2
    slow_output_gain_lower_bound: float = -2.0
    slow_output_gain_upper_bound: float = 2.0
    w_gate_init_min: float = -1.0
    w_gate_init_max: float = 1.0
    w_gate_mutate_power: float = 0.3
    b_gate_init_min: float = -0.5
    b_gate_init_max: float = 0.5
    b_gate_mutate_power: float = 0.2
    gated_alpha_slow_init_min: float = 0.8
    gated_alpha_slow_init_max: float = 0.99
    gated_alpha_slow_mutate_power: float = 0.05
    bias_fast_init_min: float = -0.5
    bias_fast_init_max: float = 0.5
    bias_fast_mutate_power: float = 0.2
    bias_slow_init_min: float = -0.5
    bias_slow_init_max: float = 0.5
    bias_slow_mutate_power: float = 0.2
    gated_mutate_rate: float = 0.6
    content_w_key_init_min: float = -1.0
    content_w_key_init_max: float = 1.0
    content_w_key_mutate_power: float = 0.2
    content_b_key_init_min: float = -0.5
    content_b_key_init_max: float = 0.5
    content_b_key_mutate_power: float = 0.2
    content_w_query_init_min: float = -1.0
    content_w_query_init_max: float = 1.0
    content_w_query_mutate_power: float = 0.2
    content_b_query_init_min: float = -0.5
    content_b_query_init_max: float = 0.5
    content_b_query_mutate_power: float = 0.2
    content_temperature_init_min: float = 0.5
    content_temperature_init_max: float = 2.0
    content_temperature_mutate_power: float = 0.1
    content_b_match_init_min: float = -0.5
    content_b_match_init_max: float = 0.5
    content_b_match_mutate_power: float = 0.2
    content_mutate_rate: float = 0.6


@dataclass(frozen=True)
class PlasticityConfig:
    delta_w_clamp: float = 1.0


@dataclass(frozen=True)
class SpacetimeConfig:
    server_url: str = "http://127.0.0.1:3000"
    database_name: str = "neat-xor-v1"
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class AppConfig:
    task: TaskConfig = TaskConfig()
    run: RunConfig = RunConfig()
    online: OnlineConfig = OnlineConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    mutation: MutationConfig = MutationConfig()
    plasticity: PlasticityConfig = PlasticityConfig()
    spacetimedb: SpacetimeConfig = SpacetimeConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AppConfig":
        task_payload = dict(payload.get("task", {}))
        if "evaluation_delay_steps" in task_payload:
            task_payload["evaluation_delay_steps"] = _coerce_delay_steps(task_payload["evaluation_delay_steps"])
        if "curriculum_phase_1_delay_steps" in task_payload:
            task_payload["curriculum_phase_1_delay_steps"] = _coerce_delay_steps(task_payload["curriculum_phase_1_delay_steps"])
        if "curriculum_phase_2_delay_steps" in task_payload:
            task_payload["curriculum_phase_2_delay_steps"] = _coerce_delay_steps(task_payload["curriculum_phase_2_delay_steps"])
        return cls(
            task=TaskConfig(**task_payload),
            run=RunConfig(**payload.get("run", {})),
            online=OnlineConfig(**payload.get("online", {})),
            evolution=EvolutionConfig(**payload.get("evolution", {})),
            mutation=MutationConfig(**payload.get("mutation", {})),
            plasticity=PlasticityConfig(**payload.get("plasticity", {})),
            spacetimedb=SpacetimeConfig(**payload.get("spacetimedb", {})),
        )


def _deep_merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _coerce_delay_steps(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return tuple(int(part) for part in parts if part)
    if isinstance(value, Sequence):
        return tuple(int(item) for item in value)
    raise TypeError(f"Unsupported evaluation_delay_steps payload: {type(value)!r}")


def curriculum_switch_generation(task: TaskConfig) -> int:
    legacy_value = task.curriculum_phase_split_generation
    if legacy_value is not None:
        return max(0, int(legacy_value))
    return max(0, int(task.curriculum_phase_switch_generation))


def curriculum_phase_name(task: TaskConfig, generation_id: int | None = None) -> str:
    if not task.curriculum_enabled:
        return "static"
    if generation_id is None:
        return "curriculum"
    split_generation = curriculum_switch_generation(task)
    if generation_id < split_generation:
        return "phase_1"
    return "phase_2"


def resolved_evaluation_delay_steps(task: TaskConfig, generation_id: int | None = None) -> tuple[int, ...]:
    if task.curriculum_enabled and generation_id is not None:
        split_generation = curriculum_switch_generation(task)
        if generation_id < split_generation:
            phase_delay_steps = task.curriculum_phase_1_delay_steps or (5,)
        else:
            phase_delay_steps = task.curriculum_phase_2_delay_steps or task.evaluation_delay_steps or (5, 8)
        return tuple(int(delay_steps) for delay_steps in phase_delay_steps)
    if task.evaluation_delay_steps:
        return tuple(int(delay_steps) for delay_steps in task.evaluation_delay_steps)
    return (int(task.temporal_delay_steps),)


def evaluation_delay_label(task: TaskConfig, generation_id: int | None = None) -> str:
    if task.curriculum_enabled and generation_id is None:
        phase_1_label = ",".join(str(delay_steps) for delay_steps in resolved_evaluation_delay_steps(task, generation_id=0))
        phase_2_generation = curriculum_switch_generation(task)
        phase_2_label = ",".join(str(delay_steps) for delay_steps in resolved_evaluation_delay_steps(task, generation_id=phase_2_generation))
        return f"{phase_1_label}->{phase_2_label}@g{phase_2_generation}"
    return ",".join(str(delay_steps) for delay_steps in resolved_evaluation_delay_steps(task, generation_id=generation_id))


def curriculum_phase_delay_labels(task: TaskConfig) -> tuple[str, str]:
    if not task.curriculum_enabled:
        return "", ""
    phase_1_label = ",".join(str(delay_steps) for delay_steps in resolved_evaluation_delay_steps(task, generation_id=0))
    phase_2_generation = curriculum_switch_generation(task)
    phase_2_label = ",".join(
        str(delay_steps)
        for delay_steps in resolved_evaluation_delay_steps(task, generation_id=phase_2_generation)
    )
    return phase_1_label, phase_2_label


def resolved_key_value_profile(task: TaskConfig, generation_id: int | None = None) -> str:
    default_profile = str(task.key_value_profile or "kv_full").strip() or "kv_full"
    if task.curriculum_enabled and generation_id is not None:
        split_generation = curriculum_switch_generation(task)
        if generation_id < split_generation:
            if task.name == "key_value_memory":
                return str(task.curriculum_phase_1_key_value_profile or "kv_easy").strip() or "kv_easy"
            return default_profile
        if task.name == "key_value_memory":
            return str(task.curriculum_phase_2_key_value_profile or default_profile).strip() or default_profile
        return default_profile
    return default_profile


def key_value_profile_labels(task: TaskConfig) -> tuple[str, str]:
    if not task.curriculum_enabled:
        return "", ""
    phase_1_profile = resolved_key_value_profile(task, generation_id=0)
    phase_2_profile = resolved_key_value_profile(task, generation_id=curriculum_switch_generation(task))
    return phase_1_profile, phase_2_profile


def load_config(paths: Sequence[str | Path]) -> AppConfig:
    merged: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        merged = _deep_merge(merged, data)

    return AppConfig.from_dict(merged)
