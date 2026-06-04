import cv2
import torch
import numpy as np
import time

model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
prev_time = time.time()

# --- Obstacle avoidance config ---
DANGER_THRESHOLD = 180    # depth value above which pixel is "close" (0-255, higher = closer)
BLOCKED_RATIO    = 0.30   # if >30% of zone pixels are "close", zone is blocked
TOP_CROP         = 0.30   # ignore top 30% of frame (sky / ceiling)
SMOOTH_WINDOW    = 5      # frames to average for temporal smoothing

from collections import deque
depth_buffer = deque(maxlen=SMOOTH_WINDOW)


def analyze_zones(depth_norm, h, w):
    """Split ROI into left/center/right and return obstacle score per zone."""
    top = int(h * TOP_CROP)
    roi = depth_norm[top:, :]

    zones = {
        "LEFT":   roi[:, :w // 3],
        "CENTER": roi[:, w // 3: 2 * w // 3],
        "RIGHT":  roi[:, 2 * w // 3:]
    }
    return {name: np.mean(zone > DANGER_THRESHOLD) for name, zone in zones.items()}


def decide_action(scores):
    blocked = {k: v > BLOCKED_RATIO for k, v in scores.items()}

    if not blocked["CENTER"]:
        return "GO FORWARD",  (0, 255, 0)
    elif not blocked["RIGHT"]:
        return "TURN RIGHT", (0, 200, 255)
    elif not blocked["LEFT"]:
        return "TURN LEFT",  (0, 200, 255)
    else:
        return "STOP",       (0, 0, 255)


def draw_zone_overlays(vis, scores, h, w):
    """Draw colored zone boxes on the depth visualization."""
    top = int(h * TOP_CROP)
    zone_bounds = [0, w // 3, 2 * w // 3, w]
    zone_names  = ["LEFT", "CENTER", "RIGHT"]
    colors      = {True: (0, 0, 200), False: (0, 200, 0)}  # red=blocked, green=clear

    for i, name in enumerate(zone_names):
        x1, x2 = zone_bounds[i], zone_bounds[i + 1]
        blocked = scores[name] > BLOCKED_RATIO
        color   = colors[blocked]

        cv2.rectangle(vis, (x1, top), (x2, h - 1), color, 2)
        cv2.putText(vis, f"{name}", (x1 + 5, top + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis, f"{scores[name]:.2f}", (x1 + 5, top + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth = np.nan_to_num(depth)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # --- Temporal smoothing ---
    depth_buffer.append(depth.astype(np.float32))
    depth_smooth = np.mean(depth_buffer, axis=0).astype(np.uint8)

    # --- Obstacle logic ---
    scores         = analyze_zones(depth_smooth, h, w)
    action, color  = decide_action(scores)

    # --- Visualization ---
    depth_colored = cv2.applyColorMap(depth_smooth, cv2.COLORMAP_MAGMA)
    draw_zone_overlays(depth_colored, scores, h, w)

    combined = cv2.hconcat([frame, depth_colored])

    # Action banner
    cv2.rectangle(combined, (0, h - 50), (w * 2, h), (30, 30, 30), -1)
    cv2.putText(combined, f"Action: {action}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("RGB | Depth + Obstacle Avoidance", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()