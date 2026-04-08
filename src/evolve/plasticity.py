from __future__ import annotations

STATEFUL_V2_VARIANTS = frozenset({"stateful_v2"})
STATEFUL_V2_GATED_VARIANTS = frozenset({"stateful_v2_gated"})
CONTENT_GATED_VARIANTS = frozenset({"content_gated"})
STATEFUL_V3_KV_VARIANTS = frozenset({"stateful_v3_kv"})
STATEFUL_V4_SLOTS_VARIANTS = frozenset({"stateful_v4_slots"})
STATEFUL_V5_ADDRESSED_SLOTS_VARIANTS = frozenset({"stateful_v5_addressed_slots"})
STATEFUL_V6_DELTA_MEMORY_VARIANTS = frozenset({"stateful_v6_delta_memory"})
PLASTIC_HEBB_VARIANTS = frozenset({"stateful_plastic", "stateful_plastic_hebb"})
PLASTIC_AD_FIXED_D_VARIANTS = {
    "stateful_plastic_ad_d0": 0.0,
    "stateful_plastic_ad_d005": -0.05,
    "stateful_plastic_ad_d01": -0.1,
    "stateful_plastic_ad_d02": -0.2,
}
PLASTIC_AD_NARROW_VARIANTS = frozenset({"stateful_plastic_ad_narrow"})
PLASTIC_AD_VARIANTS = frozenset({"stateful_plastic_ad", *PLASTIC_AD_FIXED_D_VARIANTS, *PLASTIC_AD_NARROW_VARIANTS})
PLASTIC_VARIANTS = frozenset((*PLASTIC_HEBB_VARIANTS, *PLASTIC_AD_VARIANTS))


def plastic_mode_for_variant(variant: str) -> str | None:
    if variant in PLASTIC_HEBB_VARIANTS:
        return "hebb"
    if variant in PLASTIC_AD_VARIANTS:
        return "ad"
    return None


def is_stateful_v2_variant(variant: str) -> bool:
    return variant in STATEFUL_V2_VARIANTS


def is_stateful_v2_gated_variant(variant: str) -> bool:
    return variant in STATEFUL_V2_GATED_VARIANTS


def is_content_gated_variant(variant: str) -> bool:
    return variant in CONTENT_GATED_VARIANTS


def is_stateful_v3_kv_variant(variant: str) -> bool:
    return variant in STATEFUL_V3_KV_VARIANTS


def is_stateful_v4_slots_variant(variant: str) -> bool:
    return variant in STATEFUL_V4_SLOTS_VARIANTS


def is_stateful_v5_addressed_slots_variant(variant: str) -> bool:
    return variant in STATEFUL_V5_ADDRESSED_SLOTS_VARIANTS


def is_stateful_v6_delta_memory_variant(variant: str) -> bool:
    return variant in STATEFUL_V6_DELTA_MEMORY_VARIANTS


def plastic_fixed_d_for_variant(variant: str) -> float | None:
    return PLASTIC_AD_FIXED_D_VARIANTS.get(variant)


def plastic_d_bounds_for_variant(
    variant: str,
    *,
    default_lower_bound: float,
    default_upper_bound: float,
) -> tuple[float, float]:
    fixed_plastic_d = plastic_fixed_d_for_variant(variant)
    if fixed_plastic_d is not None:
        return fixed_plastic_d, fixed_plastic_d
    if variant in PLASTIC_AD_NARROW_VARIANTS:
        return -0.1, 0.0
    return float(default_lower_bound), float(default_upper_bound)


def plastic_d_init_for_variant(
    variant: str,
    *,
    default_mean: float,
    default_std: float,
) -> tuple[float, float]:
    if variant in PLASTIC_AD_NARROW_VARIANTS:
        return -0.02, 0.01
    return float(default_mean), float(default_std)


def plastic_d_mutation_for_variant(
    variant: str,
    *,
    default_power: float,
    default_rate: float,
    default_replace_rate: float,
) -> tuple[float, float, float]:
    if variant in PLASTIC_AD_NARROW_VARIANTS:
        return 0.01, float(default_rate), 0.01
    return float(default_power), float(default_rate), float(default_replace_rate)


def is_plastic_variant(variant: str) -> bool:
    return plastic_mode_for_variant(variant) is not None
