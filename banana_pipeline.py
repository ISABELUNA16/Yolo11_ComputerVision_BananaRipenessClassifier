#Pipeline en dos etapas: detectar plátano -> recorte cuadrado -> clasificar madurez
from __future__ import annotations

from PIL import Image, ImageOps
from ultralytics import YOLO

BANANA_NAMES = {"banana", "platano", "plátano"}


class BananaPipeline:
    def __init__(
        self,
        det_weights: str = "yolo11n.pt",   # detector COCO (se descarga solo la 1.ª vez)
        cls_weights: str = "runs/classify/BananaRipenessClasifier/yolo_cls_experiment/weights/best.pt",
        det_conf: float = 0.35,            # umbral del detector: ajustarlo con 05_evaluate_gate.py
        pad: float = 0.10,                 # margen del 10 % alrededor de la caja
        min_box_frac: float = 0.01,        # descarta cajas < 1 % del área de la imagen
        fill=(255, 255, 255),              # relleno blanco: el dataset tiene fondo claro
    ):
        self.det = YOLO(det_weights)
        self.cls = YOLO(cls_weights)
        self.det_conf = det_conf
        self.pad = pad
        self.min_box_frac = min_box_frac
        self.fill = fill

        # Buscar la clase 'banana' por nombre (en COCO es la 46)
        self.banana_ids = [k for k, v in self.det.names.items() if v.lower() in BANANA_NAMES]
        if not self.banana_ids:
            raise ValueError(f"El detector no tiene clase 'banana'. Clases: {self.det.names}")

    def square_crop(self, img: Image.Image, box) -> Image.Image:
        # Recorte CUADRADO
        W, H = img.size
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = int(round(max(x2 - x1, y2 - y1) * (1 + 2 * self.pad)))
        left, top = int(round(cx - side / 2)), int(round(cy - side / 2))

        canvas = Image.new("RGB", (side, side), self.fill)
        region = img.crop((max(left, 0), max(top, 0), min(left + side, W), min(top + side, H)))
        canvas.paste(region, (max(left, 0) - left, max(top, 0) - top))
        return canvas

    def predict(self, image: Image.Image) -> list[dict]:
        """Devuelve una lista vacía si no hay plátanos (=> la imagen se rechaza)."""
        image = ImageOps.exif_transpose(image).convert("RGB")  # fotos de celular rotadas
        W, H = image.size

        # Etapa 1: detección (solo la clase banana)
        det = self.det.predict(image, conf=self.det_conf, classes=self.banana_ids, verbose=False)[0]

        boxes, crops = [], []
        for (x1, y1, x2, y2), conf in zip(det.boxes.xyxy.tolist(), det.boxes.conf.tolist()):
            if (x2 - x1) * (y2 - y1) / (W * H) < self.min_box_frac:
                continue
            boxes.append(((int(x1), int(y1), int(x2), int(y2)), float(conf)))
            crops.append(self.square_crop(image, (x1, y1, x2, y2)))

        if not crops:
            return []

        # Etapa 2: clasificación de madurez (todos los recortes en un solo lote)
        cls_out = self.cls.predict(crops, verbose=False)

        results = []
        for (box, det_conf), crop, r in zip(boxes, crops, cls_out):
            probs = r.probs.data.cpu().numpy()
            top = int(probs.argmax())
            results.append({
                "box": box,
                "det_conf": det_conf,
                "crop": crop,
                "label": self.cls.names[top],
                "cls_conf": float(probs[top]),
                "probs": {self.cls.names[i]: float(p) for i, p in enumerate(probs)},
            })
        return results