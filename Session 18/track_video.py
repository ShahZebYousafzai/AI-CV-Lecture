"""
Session 18 — Multi-Object Tracking Inference Script
=====================================================
Runs ByteTrack or DeepSORT on a video file using YOLOv8 as the detector.

Usage examples:
  python track_video.py --video input.mp4
  python track_video.py --video input.mp4 --tracker deepsort
  python track_video.py --video input.mp4 --tracker bytetrack --classes 0 2
  python track_video.py --video input.mp4 --conf 0.35 --no-trace --output my_result.mp4
  python track_video.py --video 0                          # webcam
  python track_video.py --video input.mp4 --show          # display while running

Dependencies:
  pip install ultralytics supervision deep-sort-realtime opencv-python
"""

import argparse
import time
import sys
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ── Lazy imports (only load what's needed) ────────────────────────────────────

def _import_bytetrack():
    try:
        import supervision as sv
        from ultralytics import YOLO
        return sv, YOLO
    except ImportError:
        sys.exit("Missing packages. Run: pip install ultralytics supervision")

def _import_deepsort():
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        from ultralytics import YOLO
        return DeepSort, YOLO
    except ImportError:
        sys.exit("Missing packages. Run: pip install ultralytics deep-sort-realtime")


# ── COCO class names (first 80) ───────────────────────────────────────────────

COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _id_colour(track_id: int):
    """Return a consistent BGR colour for a given track ID."""
    np.random.seed(track_id * 37 + 13)
    r, g, b = np.random.randint(80, 240, 3)
    return int(b), int(g), int(r)   # OpenCV is BGR


# ── Annotator helpers ─────────────────────────────────────────────────────────

def _draw_box_label(frame, x1, y1, x2, y2, tid, label, colour, thickness=2):
    """Draw a bounding box, filled label tag, and ID."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    tag = f"#{tid}  {label}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty), colour, -1)
    cv2.putText(frame, tag, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_trace(frame, trace: list, colour, thickness=2):
    """Draw the trajectory trail for a track."""
    pts = np.array(trace, dtype=np.int32)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in colour)
        cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), c, thickness)


def _draw_hud(frame, tracker_name, frame_id, fps, n_tracks, width):
    """Overlay a translucent HUD bar at the top of the frame."""
    bar_h = 38
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    text = (f"{tracker_name}  |  Frame {frame_id:04d}  |"
            f"  Active tracks: {n_tracks}  |  {fps:.1f} FPS")
    cv2.putText(frame, text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 1, cv2.LINE_AA)


# ── ByteTrack runner ──────────────────────────────────────────────────────────

def run_bytetrack(args):
    sv, YOLO = _import_bytetrack()
    import supervision as sv_mod   # needed for version check

    model   = YOLO(args.model)
    tracker = sv.ByteTrack(
        track_activation_threshold=args.conf,
        lost_track_buffer=args.lost_frames,
        minimum_matching_threshold=0.8,
        frame_rate=args.fps or 30,
    )
    if args.trace:
        trace_ann = sv.TraceAnnotator(trace_length=args.trace_len, thickness=2)

    cap    = _open_source(args.video)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _make_writer(args.output, fps, W, H) if args.output else None

    classes  = args.classes or None   # None = all classes
    stats    = _Stats()
    frame_id = 0

    print(f"\n[ByteTrack]  model={args.model}  conf={args.conf}"
          f"  classes={classes or 'all'}  source={args.video}\n")

    while cap.isOpened():
        if args.max_frames and frame_id >= args.max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        t0      = time.perf_counter()
        results = model(frame, verbose=False,
                        conf=args.conf, classes=classes)[0]
        dets    = sv.Detections.from_ultralytics(results)
        dets    = tracker.update_with_detections(dets)
        elapsed = time.perf_counter() - t0

        stats.update(elapsed, len(dets))

        # ── Annotate ──
        annotated = frame.copy()

        if args.trace and len(dets) > 0:
            annotated = trace_ann.annotate(annotated, dets)

        if dets.tracker_id is not None:
            for box, tid, cls_id, conf_val in zip(
                    dets.xyxy,
                    dets.tracker_id,
                    dets.class_id,
                    dets.confidence):
                x1, y1, x2, y2 = map(int, box)
                colour  = _id_colour(int(tid))
                cls_name = COCO_NAMES[int(cls_id)] if int(cls_id) < len(COCO_NAMES) else str(cls_id)
                _draw_box_label(annotated, x1, y1, x2, y2,
                                int(tid), f"{cls_name} {conf_val:.0%}", colour)

        n_active = len(set(dets.tracker_id.tolist())) if dets.tracker_id is not None else 0
        _draw_hud(annotated, "ByteTrack", frame_id, 1 / elapsed, n_active, W)

        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("ByteTrack", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1
        if frame_id % 30 == 0:
            print(f"  frame {frame_id:04d} | {1/elapsed:5.1f} FPS | tracks {n_active}")

    _cleanup(cap, writer, args.show)
    stats.report("ByteTrack", args.output)


# ── DeepSORT runner ───────────────────────────────────────────────────────────

def run_deepsort(args):
    DeepSort, YOLO = _import_deepsort()

    model   = YOLO(args.model)
    tracker = DeepSort(
        max_age=args.lost_frames,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.4,
        nn_budget=None,
        embedder=args.embedder,
        half=True,
        bgr=True,
    )

    cap    = _open_source(args.video)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _make_writer(args.output, fps, W, H) if args.output else None

    classes   = args.classes or None
    stats     = _Stats()
    frame_id  = 0
    histories = defaultdict(list)   # tid → list of (cx, cy) for trace

    print(f"\n[DeepSORT]  model={args.model}  conf={args.conf}"
          f"  embedder={args.embedder}  classes={classes or 'all'}  source={args.video}\n")

    while cap.isOpened():
        if args.max_frames and frame_id >= args.max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        t0      = time.perf_counter()
        results = model(frame, verbose=False,
                        conf=args.conf, classes=classes)[0]

        # Build DeepSORT input: list of ([x1, y1, w, h], conf, class_name)
        raw = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_val        = box.conf[0].item()
            cls_id          = int(box.cls[0].item())
            cls_name        = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else str(cls_id)
            raw.append(([x1, y1, x2 - x1, y2 - y1], conf_val, cls_name))

        tracks  = tracker.update_tracks(raw, frame=frame)
        elapsed = time.perf_counter() - t0

        stats.update(elapsed, sum(1 for t in tracks if t.is_confirmed()))

        # ── Annotate ──
        annotated = frame.copy()
        n_active  = 0

        for track in tracks:
            if not track.is_confirmed():
                continue
            n_active += 1
            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            colour  = _id_colour(int(tid))
            cls_name = track.det_class or "object"

            # Update trace history
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            histories[tid].append((cx, cy))
            if len(histories[tid]) > args.trace_len:
                histories[tid].pop(0)

            if args.trace and len(histories[tid]) > 1:
                _draw_trace(annotated, histories[tid], colour)

            _draw_box_label(annotated, x1, y1, x2, y2, int(tid), cls_name, colour)

        _draw_hud(annotated, "DeepSORT", frame_id, 1 / elapsed, n_active, W)

        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("DeepSORT", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1
        if frame_id % 30 == 0:
            print(f"  frame {frame_id:04d} | {1/elapsed:5.1f} FPS | tracks {n_active}")

    _cleanup(cap, writer, args.show)
    stats.report("DeepSORT", args.output)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _open_source(source):
    """Open a video file or webcam index."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video source: {source}")
    return cap


def _make_writer(output_path, fps, w, h):
    """Create an OpenCV VideoWriter."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[WARN] Could not open writer for {output_path}")
        return None
    return writer


def _cleanup(cap, writer, show):
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()


class _Stats:
    """Lightweight FPS and track counter."""
    def __init__(self):
        self.times  = []
        self.counts = []
        self.t_wall = time.perf_counter()

    def update(self, elapsed, n_tracks):
        self.times.append(elapsed)
        self.counts.append(n_tracks)

    def report(self, name, output):
        if not self.times:
            return
        avg_fps  = 1 / np.mean(self.times)
        peak_fps = 1 / np.min(self.times)
        avg_trk  = np.mean(self.counts)
        wall     = time.perf_counter() - self.t_wall
        n_frames = len(self.times)
        print("\n" + "=" * 52)
        print(f"  {name} — Run Summary")
        print("=" * 52)
        print(f"  Frames processed : {n_frames}")
        print(f"  Wall time        : {wall:.1f}s")
        print(f"  Average FPS      : {avg_fps:.1f}")
        print(f"  Peak FPS         : {peak_fps:.1f}")
        print(f"  Avg active tracks: {avg_trk:.1f}")
        if output:
            size_mb = Path(output).stat().st_size / 1e6 if Path(output).exists() else 0
            print(f"  Output saved     : {output}  ({size_mb:.1f} MB)")
        print("=" * 52 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Object Tracking inference — ByteTrack or DeepSORT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Source / output
    p.add_argument("--video",  required=True,
                   help="Path to input video, or '0' for webcam")
    p.add_argument("--output", default=None,
                   help="Path for the annotated output video (default: auto-named next to input)")

    # Tracker
    p.add_argument("--tracker", choices=["bytetrack", "deepsort"], default="bytetrack",
                   help="Tracker to use (default: bytetrack)")

    # Detector
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLOv8 model weights (default: yolov8n.pt). "
                        "Options: yolov8n/s/m/l/x.pt or a custom .pt path")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detection confidence threshold (default: 0.25)")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="COCO class IDs to track, e.g. --classes 0 2 (person, car). "
                        "Default: all classes")

    # Tracker settings
    p.add_argument("--lost-frames", type=int, default=30,
                   help="Frames to keep a lost track alive before deleting (default: 30)")
    p.add_argument("--fps", type=float, default=None,
                   help="Video FPS override (auto-detected if omitted)")

    # DeepSORT-specific
    p.add_argument("--embedder", default="mobilenet",
                   choices=["mobilenet", "torchreid", "clip_RN50", "clip_RN101",
                             "clip_ViT-B/32", "clip_ViT-B/16"],
                   help="DeepSORT ReID backbone (default: mobilenet)")

    # Visualisation
    p.add_argument("--show",     action="store_true",
                   help="Display tracking window in real time (press Q to quit)")
    p.add_argument("--no-trace", dest="trace", action="store_false",
                   help="Disable trajectory trail annotation")
    p.add_argument("--trace-len", type=int, default=40,
                   help="Number of past frames shown in the trace trail (default: 40)")
    p.set_defaults(trace=True)

    # Limits
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many frames (useful for quick tests)")

    return p.parse_args()


def _auto_output(video_path: str, tracker: str) -> str:
    """Generate a default output path next to the input file."""
    if video_path.isdigit():
        return f"webcam_{tracker}.mp4"
    p = Path(video_path)
    return str(p.parent / f"{p.stem}_{tracker}{p.suffix}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.output is None:
        args.output = _auto_output(args.video, args.tracker)

    print(f"\n{'='*52}")
    print(f"  Session 18 — Multi-Object Tracking")
    print(f"  Tracker  : {args.tracker.upper()}")
    print(f"  Source   : {args.video}")
    print(f"  Output   : {args.output}")
    print(f"  Model    : {args.model}")
    print(f"  Classes  : {args.classes or 'all'}")
    print(f"  Trace    : {'on' if args.trace else 'off'}")
    print(f"{'='*52}\n")

    if args.tracker == "bytetrack":
        run_bytetrack(args)
    else:
        run_deepsort(args)


if __name__ == "__main__":
    main()
