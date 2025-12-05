import cv2
import numpy as np
from ultralytics import YOLO
import mss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import random

# =====================================================
# === CONFIGURAÇÕES GERAIS ============================
# =====================================================
model_path_yolo = r"C:/Users/Gabriel/Desktop/Trabalho de Deep Learning/pipeline v2/1 YOLO11 seg carta/runs/segment/train/weights/best.pt"
model_path_orientation = r"C:/Users/Gabriel/Desktop/Trabalho de Deep Learning/pipeline v2/2 PyTorch cls/orientationnet_best.pth"
model_path_digits = r"C:/Users/Gabriel/Desktop/Trabalho de Deep Learning/pipeline v2/3 YOLO11 det dígitos/runs/detect/train/weights/best.pt"

source = 0
conf_thr_seg = 0.5
conf_thr_digits = 0.25
imgsz_seg = 448
imgsz_digits = 640
show_size = (840, 560)
crop_size = (400, 250)

# =====================================================
# === CORES ALEATÓRIAS PARA 25 CLASSES ===============
# =====================================================
NUM_CLASSES = 25
colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(NUM_CLASSES)
]

# =====================================================
# === ORIENTATIONNET ==================================
# =====================================================
class OrientationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# =====================================================
# === CARREGAR MODELOS ================================
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Dispositivo ativo:", device)

orientation_model = OrientationNet().to(device)
orientation_model.load_state_dict(torch.load(model_path_orientation, map_location=device))
orientation_model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

angle_map = {0: 0, 2: 90, 1: 180, 3: 270}

yolo_seg = YOLO(model_path_yolo)
yolo_digits = YOLO(model_path_digits)

# =====================================================
# === CAPTURA =========================================
# =====================================================
if source == "screen":
    sct = mss.mss()
    monitor = sct.monitors[1]
else:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("❌ Erro ao iniciar webcam.")

# =====================================================
# === FUNÇÃO DE CROP ROTACIONADO ======================
# =====================================================
def crop_min_area_rect(image, rect):
    (cx, cy), (w, h), angle = rect
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(image.shape[1], int(cx + w / 2))
    y2 = min(image.shape[0], int(cy + h / 2))
    return rotated[y1:y2, x1:x2]

# =====================================================
# === LOOP PRINCIPAL ==================================
# =====================================================
while True:
    frame_start = time.time()

    if source == "screen":
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        ret, frame = cap.read()
        if not ret:
            break

    display = frame.copy()
    crops = []

    # =================================================
    # YOLO SEGMENTAÇÃO
    # =================================================
    t0 = time.time()
    results = yolo_seg(frame, conf=conf_thr_seg, imgsz=imgsz_seg, verbose=False)
    seg_ms = (time.time() - t0) * 1000

    r = results[0]
    if r.masks is not None:
        for poly in r.masks.xy:

            pts = np.array(poly, dtype=np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(int)

            # Desenha a carta na imagem principal
            cv2.polylines(display, [box], True, (255, 0, 0), 2)

            crop = crop_min_area_rect(frame, rect)
            if crop.size == 0:
                continue

            # =================================================
            # ORIENTATION NET
            # =================================================
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            x = transform(pil).unsqueeze(0).to(device)

            t1 = time.time()
            with torch.no_grad():
                out = orientation_model(x)
                pred = torch.argmax(out, dim=1).item()
                angle = angle_map.get(pred, 0)
            cls_ms = (time.time() - t1) * 1000

            if angle == 90:
                corrected = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                corrected = cv2.rotate(crop, cv2.ROTATE_180)
            elif angle == 270:
                corrected = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                corrected = crop

            # =================================================
            # YOLO DÍGITOS NO CROP CORRIGIDO
            # =================================================
            digit_results = yolo_digits(corrected, conf=conf_thr_digits, imgsz=imgsz_digits, iou=0.7, verbose=False)
            dr = digit_results[0]

            code_str = ""  # string final do código lido

            if dr.boxes is not None and len(dr.boxes) > 0:
                boxes = dr.boxes
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                names = yolo_digits.names

                # lista para ordenar os dígitos pelo eixo x (esquerda -> direita)
                digit_items = []

                for (x1, y1, x2, y2), cid in zip(xyxy, cls_ids):
                    color = colors[cid % NUM_CLASSES]

                    # desenha bbox no crop corrigido
                    cv2.rectangle(corrected, (x1, y1), (x2, y2), color, 2)

                    if isinstance(names, dict):
                        name = names.get(cid, str(cid))
                    else:
                        name = names[cid] if cid < len(names) else str(cid)

                    name = str(name)

                    cv2.putText(
                        corrected, name,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA
                    )

                    # centro em x pra ordenar, mais o caractere
                    cx = (x1 + x2) / 2.0
                    digit_items.append((cx, name))

                # ordenar da esquerda para a direita e montar string
                digit_items.sort(key=lambda t: t[0])
                code_str = "".join(ch for _, ch in digit_items)

                # opcional: também escreve o código no crop corrigido (embaixo)
                if code_str:
                    cv2.putText(
                        corrected, code_str,
                        (10, corrected.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA
                    )

            print(f"SEG: {seg_ms:.2f} ms | CLS: {cls_ms:.2f} ms")
            crops.append(corrected)

    frame_total_ms = (time.time() - frame_start) * 1000
    print(f"FRAME TOTAL: {frame_total_ms:.2f} ms")

    cv2.imshow("YOLO11 SEG + OrientationNet", cv2.resize(display, show_size))

    if crops:
        cv2.imshow("Crops Corrigidos", np.hstack([cv2.resize(c, crop_size) for c in crops]))
    else:
        cv2.imshow("Crops Corrigidos", np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if source != "screen":
    cap.release()

cv2.destroyAllWindows()
