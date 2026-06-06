import cv2
import torch
import numpy as np

model_type = "DPT_Large"   # faster for CPU

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

cap = cv2.VideoCapture(3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = depth.astype("uint8")

    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)

    combined = cv2.hconcat([frame, depth_colored])

    cv2.imshow("RGB | Depth", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()