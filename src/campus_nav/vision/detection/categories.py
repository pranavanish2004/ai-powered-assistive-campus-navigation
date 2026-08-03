"""
Object category vocabulary for navigation-relevant obstacle detection.

Design rationale
-----------------
This module is deliberately pure data: an enum and a couple of lookup
dicts, with zero imports beyond the standard library. That is what allows
`interfaces/detection_interface.py` to depend on it directly — the same
pattern already established by `interfaces/navigation_interface.py`
depending on `navigation.routing.types` (also pure data). A detection
*backend* (YOLOv8, a future model) maps its own raw class vocabulary onto
`ObjectCategory` via `COCO_LABEL_TO_CATEGORY`, so the rest of the vision
pipeline and the Safety Engine never need to know which underlying model
produced a detection.

Categories are chosen to match the original project specification's
"Object Detection Policy": most everyday objects should be silently
ignored unless they are blocking the user's path; a small set (doors,
ramps, stairs) are always worth announcing regardless of position, because
they are inherently navigation-relevant.
"""

from __future__ import annotations

from enum import Enum


class ObjectCategory(str, Enum):
    """Normalized object category, independent of any specific detection
    model's raw label vocabulary."""

    PERSON = "person"
    VEHICLE = "vehicle"
    DOOR = "door"
    STAIRS = "stairs"
    RAMP = "ramp"
    BENCH = "bench"
    POLE = "pole"
    TREE = "tree"
    ANIMAL = "animal"
    FURNITURE = "furniture"
    UNKNOWN = "unknown"


# Always-relevant categories: per the original object detection policy,
# these are worth announcing regardless of whether they are technically
# "blocking" the path, because they represent navigation-critical
# infrastructure (a door or a flight of stairs matters even off to one
# side of the walking path).
ALWAYS_ANNOUNCE_CATEGORIES: frozenset[ObjectCategory] = frozenset(
    {ObjectCategory.DOOR, ObjectCategory.STAIRS, ObjectCategory.RAMP}
)

# Categories that are only worth mentioning when they are actually blocking
# the user's path (see vision.safety.obstacle_events for the blocking
# determination). Matches the spec's "tree/bench beside path -> ignore,
# blocking path -> warn" policy.
CONDITIONAL_CATEGORIES: frozenset[ObjectCategory] = frozenset(
    {
        ObjectCategory.TREE,
        ObjectCategory.BENCH,
        ObjectCategory.POLE,
        ObjectCategory.FURNITURE,
        ObjectCategory.VEHICLE,
        ObjectCategory.ANIMAL,
    }
)

# A person is handled distinctly: "someone is approaching" rather than the
# generic "obstacle ahead" phrasing, and is evaluated by proximity/movement
# rather than purely by static blocking position.
PERSON_CATEGORIES: frozenset[ObjectCategory] = frozenset({ObjectCategory.PERSON})


# Maps common COCO class names (YOLOv8's default pretrained vocabulary) onto
# our normalized categories. Unmapped labels fall back to UNKNOWN, which is
# always treated as CONDITIONAL (only announced if blocking) — a
# conservative default that avoids the system chattering about objects it
# doesn't have a specific navigation policy for yet.
COCO_LABEL_TO_CATEGORY: dict[str, ObjectCategory] = {
    "person": ObjectCategory.PERSON,
    "bicycle": ObjectCategory.VEHICLE,
    "car": ObjectCategory.VEHICLE,
    "motorcycle": ObjectCategory.VEHICLE,
    "bus": ObjectCategory.VEHICLE,
    "truck": ObjectCategory.VEHICLE,
    "bench": ObjectCategory.BENCH,
    "chair": ObjectCategory.FURNITURE,
    "couch": ObjectCategory.FURNITURE,
    "dining table": ObjectCategory.FURNITURE,
    "potted plant": ObjectCategory.TREE,
    "dog": ObjectCategory.ANIMAL,
    "cat": ObjectCategory.ANIMAL,
    "horse": ObjectCategory.ANIMAL,
    "cow": ObjectCategory.ANIMAL,
}


def category_for_label(label: str) -> ObjectCategory:
    """Normalize a raw model class label into an ObjectCategory. Case- and
    whitespace-insensitive; unrecognized labels map to UNKNOWN rather than
    raising, since an unmapped label is expected as the underlying model's
    vocabulary evolves (e.g. a future custom-trained model)."""
    return COCO_LABEL_TO_CATEGORY.get(label.strip().lower(), ObjectCategory.UNKNOWN)


def is_always_announce(category: ObjectCategory) -> bool:
    return category in ALWAYS_ANNOUNCE_CATEGORIES


def is_conditional(category: ObjectCategory) -> bool:
    """True for categories silently ignored unless blocking the path,
    including UNKNOWN (see module docstring)."""
    return category in CONDITIONAL_CATEGORIES or category is ObjectCategory.UNKNOWN
