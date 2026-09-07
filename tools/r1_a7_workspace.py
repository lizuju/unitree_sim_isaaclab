import math


OFFSET_NORM_MAX_M = 0.06
SHOULDER_EXTENSION_MAX_M = 0.005
READY_SHOULDER_TO_WRIST_M = {
    "left": (0.30382918, 0.17215110, -0.21716116),
    "right": (0.28622779, -0.18800468, -0.25568381),
}


def vector_norm(values):
    return math.sqrt(sum(value * value for value in values))


def workspace_margins(offset, side):
    if len(offset) != 3 or side not in READY_SHOULDER_TO_WRIST_M:
        raise ValueError("Expected one left or right 3D wrist offset")
    if not all(math.isfinite(value) for value in offset):
        raise ValueError("Wrist offset must be finite")
    ready_vector = READY_SHOULDER_TO_WRIST_M[side]
    ready_radius = vector_norm(ready_vector)
    return (
        OFFSET_NORM_MAX_M - vector_norm(offset),
        ready_radius
        + SHOULDER_EXTENSION_MAX_M
        - vector_norm([ready_vector[index] + offset[index] for index in range(3)]),
    )


def is_offset_in_workspace(offset, side, tolerance=1.0e-9):
    return min(workspace_margins(offset, side)) >= -tolerance


def constrain_offset(offset, side):
    values = [float(value) for value in offset]
    if is_offset_in_workspace(values, side):
        return values, False
    norm = vector_norm(values)
    scale = min(1.0, OFFSET_NORM_MAX_M / norm) if norm else 1.0
    ready_vector = READY_SHOULDER_TO_WRIST_M[side]
    a = sum(value * value for value in values)
    b = 2.0 * sum(ready_vector[index] * values[index] for index in range(3))
    ready_limit = vector_norm(ready_vector) + SHOULDER_EXTENSION_MAX_M
    c = sum(value * value for value in ready_vector) - ready_limit * ready_limit
    discriminant = max(0.0, b * b - 4.0 * a * c)
    extension_scale = (-b + math.sqrt(discriminant)) / (2.0 * a)
    scale = min(scale, extension_scale)
    return [value * max(0.0, scale) for value in values], True


def constrain_dual_offset(offset):
    if len(offset) != 6:
        raise ValueError("Expected left and right 3D wrist offsets")
    left, left_limited = constrain_offset(offset[:3], "left")
    right, right_limited = constrain_offset(offset[3:], "right")
    return left + right, left_limited or right_limited


def is_dual_offset_in_workspace(offset, tolerance=1.0e-9):
    return (
        len(offset) == 6
        and is_offset_in_workspace(offset[:3], "left", tolerance)
        and is_offset_in_workspace(offset[3:], "right", tolerance)
    )
