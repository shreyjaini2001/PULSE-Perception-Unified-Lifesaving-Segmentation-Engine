"""Rule-of-Nines TBSA estimation from burn-tagged wounds (S6)."""

from __future__ import annotations

from typing import Iterable

from .victim import WoundRegion

BURN_LABEL_KEYWORDS = ("burn", "blister", "soot", "singed")

# Anatomical group -> max % TBSA for that group (adult rule of nines)
_GROUP_CAP = {
    "head": 9.0,
    "left_arm": 9.0,
    "right_arm": 9.0,
    "anterior_trunk": 18.0,
    "posterior_trunk": 18.0,
    "left_leg": 18.0,
    "right_leg": 18.0,
    "groin": 1.0,
}

# Underscore body_location -> (group, fraction of group for this segment)
_REGION_GROUP: dict[str, tuple[str, float]] = {
    "head": ("head", 1.0),
    "face": ("head", 0.5),
    "neck": ("head", 0.35),
    "chest": ("anterior_trunk", 0.5),
    "abdomen": ("anterior_trunk", 0.5),
    "left_torso": ("anterior_trunk", 0.35),
    "right_torso": ("anterior_trunk", 0.35),
    "groin": ("groin", 1.0),
    "back": ("posterior_trunk", 1.0),
    "left_shoulder": ("left_arm", 0.2),
    "left_upper_arm": ("left_arm", 0.45),
    "left_forearm": ("left_arm", 0.35),
    "left_hand": ("left_arm", 0.15),
    "right_shoulder": ("right_arm", 0.2),
    "right_upper_arm": ("right_arm", 0.45),
    "right_forearm": ("right_arm", 0.35),
    "right_hand": ("right_arm", 0.15),
    "left_thigh": ("left_leg", 0.5),
    "left_knee": ("left_leg", 0.1),
    "left_shin": ("left_leg", 0.4),
    "left_ankle": ("left_leg", 0.08),
    "left_foot": ("left_leg", 0.12),
    "right_thigh": ("right_leg", 0.5),
    "right_knee": ("right_leg", 0.1),
    "right_shin": ("right_leg", 0.4),
    "right_ankle": ("right_leg", 0.08),
    "right_foot": ("right_leg", 0.12),
}


def _is_burn_wound(w: WoundRegion) -> bool:
    lbl = (w.label or "").lower()
    return any(k in lbl for k in BURN_LABEL_KEYWORDS)


def estimate_tbsa_percent(wounds: Iterable[WoundRegion]) -> float:
    """Approximate TBSA% from burn-class wounds and body_location. Capped at 95%."""
    group_used: dict[str, float] = {g: 0.0 for g in _GROUP_CAP}
    for w in wounds:
        if not _is_burn_wound(w):
            continue
        loc = (w.body_location or "unknown").lower().replace(" ", "_")
        if loc not in _REGION_GROUP:
            continue
        grp, frac = _REGION_GROUP[loc]
        cap = _GROUP_CAP[grp]
        add = min(cap * frac, cap - group_used[grp])
        if add > 0:
            group_used[grp] += add
    total = min(95.0, sum(group_used.values()))
    return round(total, 1)
