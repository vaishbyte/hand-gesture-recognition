"""Synthetic tests for the gesture logic (no webcam needed)."""
import math
import numpy as np
from hand_gesture import fingers_up, classify, pinch_ratio

WRIST = (300.0, 400.0)
MCP = {1: (270.0, 300.0), 2: (300.0, 295.0), 3: (330.0, 300.0), 4: (358.0, 312.0)}
FINGER_SLOTS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]


def build(fingers, thumb, pinch=False):
    """fingers: 4 booleans (index..pinky). thumb: bool extended."""
    p = [None] * 21
    p[0] = WRIST
    for k, (mcp_i, pip_i, dip_i, tip_i) in enumerate(FINGER_SLOTS, start=1):
        mx, my = MCP[k]
        p[mcp_i] = (mx, my)
        if fingers[k - 1]:
            p[pip_i] = (mx, my - 42)
            p[dip_i] = (mx, my - 70)
            p[tip_i] = (mx, my - 95)
        else:
            p[pip_i] = (mx, my - 40)
            p[dip_i] = (mx + 4, my - 12)
            p[tip_i] = (mx + 6, my + 16)
    p[1] = (272.0, 378.0)
    if thumb:
        p[2], p[3], p[4] = (243.0, 356.0), (216.0, 332.0), (196.0, 312.0)
    else:
        p[2], p[3], p[4] = (252.0, 352.0), (250.0, 338.0), (286.0, 330.0)
    if pinch:                       # thumb tip brought to the index tip
        p[4] = (p[8][0] + 6, p[8][1] + 6)
        p[3] = (p[4][0] - 26, p[4][1] + 26)
    return np.array(p, dtype=float)


def rotate(pts, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    o = np.array(WRIST)
    return np.array([[(x - o[0]) * c - (y - o[1]) * s + o[0],
                      (x - o[0]) * s + (y - o[1]) * c + o[1]] for x, y in pts])


def scale(pts, k):
    o = np.array(WRIST)
    return (pts - o) * k + o


CASES = [
    ("FIST",       build([0, 0, 0, 0], False), [False, False, False, False, False]),
    ("OPEN PALM",  build([1, 1, 1, 1], True),  [True, True, True, True, True]),
    ("POINT",      build([1, 0, 0, 0], False), [False, True, False, False, False]),
    ("PEACE",      build([1, 1, 0, 0], False), [False, True, True, False, False]),
    ("THREE",      build([1, 1, 1, 0], False), [False, True, True, True, False]),
    ("FOUR",       build([1, 1, 1, 1], False), [False, True, True, True, True]),
    ("THUMBS UP",  build([0, 0, 0, 0], True),  [True, False, False, False, False]),
    ("CALL ME",    build([0, 0, 0, 1], True),  [True, False, False, False, True]),
    ("ROCK",       build([1, 0, 0, 1], False), [False, True, False, False, True]),
    ("I LOVE YOU", build([1, 0, 0, 1], True),  [True, True, False, False, True]),
    ("OK",         build([0, 1, 1, 1], True, pinch=True), None),
]

fails = 0
print(f"{'expected':<12} {'state':<34} {'got':<12} result")
print("-" * 74)
for expected, pts, want_state in CASES:
    st = fingers_up(pts)
    got = classify(st, pts)
    ok = got == expected and (want_state is None or st == want_state)
    fails += not ok
    print(f"{expected:<12} {str([int(b) for b in st]):<34} {got:<12} "
          f"{'OK' if ok else 'FAIL'}")

print("\nRotation invariance (PEACE, rotated about the wrist)")
base = build([1, 1, 0, 0], False)
for d in (0, 30, 60, 90, 135, 180, 250, 315):
    got = classify(fingers_up(rotate(base, d)), rotate(base, d))
    ok = got == "PEACE"
    fails += not ok
    print(f"  {d:>3} deg -> {got:<12} {'OK' if ok else 'FAIL'}")

print("\nScale invariance (OK sign, hand nearer/further from camera)")
base = build([0, 1, 1, 1], True, pinch=True)
for k in (0.4, 0.7, 1.0, 1.6, 2.5):
    s = scale(base, k)
    got = classify(fingers_up(s), s)
    ok = got == "OK"
    fails += not ok
    print(f"  x{k:<4} ratio={pinch_ratio(s):.3f} -> {got:<12} "
          f"{'OK' if ok else 'FAIL'}")

print("\nFAILURES:", fails)