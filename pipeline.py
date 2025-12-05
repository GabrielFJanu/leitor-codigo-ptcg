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
model_path_yolo = r"weights/yolo11seg_best.pt"
model_path_orientation = r"weights/orientationnet_best.pth"
model_path_digits = r"weights/yolo11det_best.pt"

source = 0
conf_thr_seg = 0.5
conf_thr_digits = 0.5
imgsz_seg = 448
imgsz_digits = 640
show_size = (840, 560)

# arquivo de log (id,codigo)
LOG_PATH = "card_codes_log.csv"

# =====================================================
# === CORES ALEATÓRIAS PARA 24 CLASSES ===============
# =====================================================
NUM_CLASSES = 24
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
# === TRACKING SIMPLES POR CENTROIDE (COM TTL) =======
# =====================================================
def get_or_assign_card_id(cx, cy, tracks, frame_idx, next_id,
                          max_dist=100, max_age=15):
    # Remover tracks muito antigos
    dead_ids = []
    for cid, info in tracks.items():
        if frame_idx - info["last_seen"] > max_age:
            dead_ids.append(cid)
    for cid in dead_ids:
        del tracks[cid]

    best_id = None
    best_dist2 = max_dist * max_dist

    for cid, info in tracks.items():
        dx = cx - info["cx"]
        dy = cy - info["cy"]
        d2 = dx * dx + dy * dy
        if d2 < best_dist2:
            best_dist2 = d2
            best_id = cid

    if best_id is not None:
        tracks[best_id]["cx"] = cx
        tracks[best_id]["cy"] = cy
        tracks[best_id]["last_seen"] = frame_idx
        return best_id, next_id

    # Cria novo ID
    cid = next_id
    tracks[cid] = {
        "cx": cx,
        "cy": cy,
        "best_code": None,
        "best_prob": 0.0,
        "last_seen": frame_idx,
    }
    return cid, next_id + 1

# =====================================================
# === LOOP PRINCIPAL ==================================
# =====================================================

tracks = {}         # ID -> {cx, cy, best_code, best_prob, last_seen}
next_card_id = 0    # próximo ID disponível
frame_idx = 0       # contador de frames

# guarda códigos já registrados por ID -> set(códigos)
seen_codes = {}

# abre CSV em append; sem cabeçalho, só "id,codigo"
log_file = open(LOG_PATH, "a", buffering=1, encoding="utf-8")

while True:
    frame_start = time.time()
    frame_idx += 1

    if source == "screen":
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        ret, frame = cap.read()
        if not ret:
            break

    display = frame.copy()

    # =================================================
    # YOLO SEGMENTAÇÃO
    # =================================================
    results = yolo_seg(frame, conf=conf_thr_seg, imgsz=imgsz_seg, verbose=False)
    r = results[0]

    if r.masks is not None:
        masks_data = r.masks.data.cpu().numpy()   # [N, h, w]
        polys = r.masks.xy                        # polígonos
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()   # bounding boxes padrão [N,4]

        for i, pts in enumerate(polys):
            pts = np.array(pts, dtype=np.float32)

            # bounding box padrão YOLO (eixo alinhado)
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)

            # centro aproximado para tracking
            cx_card = (x1 + x2) / 2
            cy_card = (y1 + y2) / 2

            card_id, next_card_id = get_or_assign_card_id(
                cx_card, cy_card, tracks, frame_idx, next_card_id
            )

            # se ainda não temos set de códigos vistos para esse ID, cria
            if card_id not in seen_codes:
                seen_codes[card_id] = set()

            # recorte da carta a partir da bounding box normal
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ORIENTATION NET
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            x = transform(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                out = orientation_model(x)
                pred = torch.argmax(out, dim=1).item()
                angle = angle_map.get(pred, 0)

            if angle == 90:
                corrected = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                corrected = cv2.rotate(crop, cv2.ROTATE_180)
            elif angle == 270:
                corrected = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                corrected = crop

            # YOLO DIGITS
            digit_results = yolo_digits(
                corrected,
                conf=conf_thr_digits,
                imgsz=imgsz_digits,
                iou=0.8,
                verbose=False
            )
            dr = digit_results[0]

            code_str = ""
            prob_product = None

            if dr.boxes is not None and len(dr.boxes) > 0:
                boxes_d = dr.boxes
                xyxy = boxes_d.xyxy.cpu().numpy().astype(int)
                cls_ids = boxes_d.cls.cpu().numpy().astype(int)
                confs = boxes_d.conf.cpu().numpy()

                names = yolo_digits.names
                digit_items = []

                for (dx1, dy1, dx2, dy2), cid, conf in zip(xyxy, cls_ids, confs):
                    name = names[cid] if cid < len(names) else str(cid)
                    cx_digit = (dx1 + dx2) / 2
                    p_digit = float(conf)
                    p_digit = max(1e-6, min(1.0, p_digit))
                    digit_items.append((cx_digit, str(name), p_digit))

                digit_items.sort(key=lambda t: t[0])
                code_str = "".join(ch for _, ch, _ in digit_items)

                prob_product = 1.0
                for _, _, p in digit_items:
                    prob_product *= p

                if len(code_str) == 13:
                    # se for um novo melhor código E ainda não foi registrado, atualiza e loga
                    if prob_product > tracks[card_id]["best_prob"]:
                        # só registra se o código ainda não está no set daquele ID
                        if code_str not in seen_codes[card_id]:
                            seen_codes[card_id].add(code_str)
                            log_file.write(f"{card_id},{code_str}\n")

                        tracks[card_id]["best_code"] = code_str
                        tracks[card_id]["best_prob"] = prob_product

            # DESENHAR MÁSCARA (AMARELA/VERDE)
            best_code = tracks[card_id]["best_code"]
            best_prob = tracks[card_id]["best_prob"]

            mask_color = (0,255,255) if best_code is None else (0,255,0)
            pts_int = np.array(pts, dtype=np.int32)

            overlay = display.copy()
            cv2.fillPoly(overlay, [pts_int], mask_color)
            display = cv2.addWeighted(overlay, 0.25, display, 0.75, 0)

            # BOUNDING BOX PADRÃO
            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0,255,255) if best_code is None else (0,255,0),
                1
            )

            # ID da carta
            cv2.putText(
                display, f"ID {card_id}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA
            )

            # Melhor código (se quiser manter na tela)
            if best_code is not None:
                prob_percent = best_prob * 100
                cv2.putText(
                    display, f"{best_code} ({prob_percent:.1f}%)",
                    (x1, max(0, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA
                )

    # =================================================
    # TEMPO TOTAL DO FRAME (única info no console)
    # =================================================
    frame_total_ms = (time.time() - frame_start) * 1000
    print(f"FRAME TOTAL: {frame_total_ms:.2f} ms")

    cv2.imshow("YOLO11 SEG + OrientationNet", cv2.resize(display, show_size))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if source != "screen":
    cap.release()

log_file.close()
cv2.destroyAllWindows()
