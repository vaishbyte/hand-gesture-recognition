import math
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------------------------------
# Landmark index constants
# --------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (mcp, pip, tip) for the four fingers - the thumb is handled separately
FINGER_JOINTS = [
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
]

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Tuning
ANGLE_EXTENDED = 140.0   # degrees at the PIP joint above which a finger counts as straight
PINCH_RATIO = 0.35       # thumb-index gap / palm size, below which it is a pinch
DEBOUNCE_FRAMES = 5      # identical readings required before a gesture is confirmed


# --------------------------------------------------------------------------
# Geometry helpers - all take an (N, 2) array of PIXEL coordinates
# --------------------------------------------------------------------------
def dist(p, q):
    """Euclidean distance between two points."""
    return float(math.hypot(q[0] - p[0], q[1] - p[1]))


def angle_at(a, b, c):
    """Interior angle at point b, formed by the segments b->a and b->c, in degrees."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosine = max(-1.0, min(1.0, cosine))          # guard against float drift
    return math.degrees(math.acos(cosine))


def palm_size(pts):
    """Reference length on the hand itself: wrist -> middle-finger base.

    Every other measurement is divided by this, which is what makes the
    thresholds independent of how far the hand is from the camera.
    """
    return dist(pts[WRIST], pts[MIDDLE_MCP]) or 1.0


# --------------------------------------------------------------------------
# Finger state
# --------------------------------------------------------------------------
def fingers_up(pts):
    """Return [thumb, index, middle, ring, pinky] as booleans.

    Four fingers: a finger is extended when it is straight at the PIP joint
    AND its tip is further from the wrist than that joint. Using both tests
    together rejects the half-curled poses that fool either test alone.

    Thumb: extended when the tip is further from the pinky base than the IP
    joint is. A folded thumb crosses the palm towards the little finger, so
    this works for either hand and at any hand rotation - no handedness
    label needed.
    """
    state = []

    # --- thumb ---
    folded = dist(pts[THUMB_TIP], pts[PINKY_MCP]) < dist(pts[THUMB_IP], pts[PINKY_MCP])
    state.append(not folded)

    # --- four fingers ---
    for mcp, pip, tip in FINGER_JOINTS:
        straight = angle_at(pts[mcp], pts[pip], pts[tip]) > ANGLE_EXTENDED
        reaching = dist(pts[WRIST], pts[tip]) > dist(pts[WRIST], pts[pip])
        state.append(bool(straight and reaching))

    return state


def finger_angles(pts):
    """PIP-joint angle per finger, for the debug panel."""
    out = [angle_at(pts[THUMB_MCP], pts[THUMB_IP], pts[THUMB_TIP])]
    for mcp, pip, tip in FINGER_JOINTS:
        out.append(angle_at(pts[mcp], pts[pip], pts[tip]))
    return out


def pinch_ratio(pts):
    """Thumb-index gap as a fraction of palm size. Scale-invariant."""
    return dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm_size(pts)


# --------------------------------------------------------------------------
# Gesture classification
# --------------------------------------------------------------------------
def classify(state, pts):
    """Map a finger-state pattern to a gesture name."""
    thumb, index, middle, ring, pinky = state
    total = sum(state)
    ref = palm_size(pts)

    # OK sign is checked first: the thumb and index touch, so both read as
    # folded-ish and the plain pattern match would miss it.
    if pinch_ratio(pts) < PINCH_RATIO and middle and ring and pinky:
        return "OK"

    if total == 0:
        return "FIST"

    if total == 5:
        return "OPEN PALM"

    if state == [True, False, False, False, False]:
        # thumb alone - use its vertical direction relative to the wrist
        rise = pts[WRIST][1] - pts[THUMB_TIP][1]      # +ve means the tip is higher
        if rise > 0.5 * ref:
            return "THUMBS UP"
        if rise < -0.5 * ref:
            return "THUMBS DOWN"
        return "THUMB OUT"

    patterns = {
        (False, True, False, False, False): "POINT",
        (False, True, True, False, False): "PEACE",
        (False, True, True, True, False): "THREE",
        (False, True, True, True, True): "FOUR",
        (True, False, False, False, True): "CALL ME",
        (False, True, False, False, True): "ROCK",
        (True, True, False, False, False): "GUN",
    }
    return patterns.get(tuple(state), f"{total} FINGERS")


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_panel(frame, count, gesture, hand_label, angles, state, fps, show_debug):
    h, w, _ = frame.shape

    cv2.rectangle(frame, (10, 10), (330, 150), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (330, 150), (255, 255, 255), 2)

    cv2.putText(frame, str(count), (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 255, 255), 6)
    cv2.putText(frame, "FINGERS", (28, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, gesture, (150, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, hand_label, (150, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

    cv2.putText(frame, f"FPS {fps:.0f}", (w - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if show_debug and angles is not None:
        y = h - 130
        cv2.rectangle(frame, (10, y - 25), (300, h - 10), (0, 0, 0), -1)
        for i, name in enumerate(FINGER_NAMES):
            colour = (0, 255, 0) if state[i] else (120, 120, 120)
            cv2.putText(frame,
                        f"{name:<7}{angles[i]:6.1f}  {'UP' if state[i] else 'down'}",
                        (20, y + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found. Close any other app using the camera, "
              "or try cv2.VideoCapture(1).")
        return

    buf = deque(maxlen=DEBOUNCE_FRAMES)
    confirmed = "-"
    confirmed_count = 0
    prev_t = cv2.getTickCount()
    fps = 0.0
    show_debug = False

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Mirror so the video behaves like a mirror for the user.
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False            # lets MediaPipe skip an internal copy
        result = hands.process(rgb)
        rgb.flags.writeable = True

        angles = state = None
        hand_label = ""

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            info = result.multi_handedness[0]

            # The frame was mirrored before process(), so MediaPipe's label is
            # the opposite of the user's actual hand. Flip it back.
            raw = info.classification[0].label
            hand_label = "Right hand" if raw == "Left" else "Left hand"

            pts = np.array([[lm.x * w, lm.y * h] for lm in hand.landmark])

            state = fingers_up(pts)
            angles = finger_angles(pts)
            name = classify(state, pts)

            buf.append((name, sum(state)))
            if len(buf) == DEBOUNCE_FRAMES and len(set(buf)) == 1:
                confirmed, confirmed_count = buf[0]

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style())
        else:
            buf.clear()
            confirmed, confirmed_count = "-", 0

        now = cv2.getTickCount()
        fps = 0.9 * fps + 0.1 * (cv2.getTickFrequency() / (now - prev_t))
        prev_t = now

        draw_panel(frame, confirmed_count, confirmed, hand_label,
                   angles, state, fps, show_debug)

        cv2.imshow("Finger Counter + Gesture Recogniser", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()