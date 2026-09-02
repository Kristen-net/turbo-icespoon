"""YOLO 检测封装 (ultralytics 为可选依赖).

旧实现的问题: YOLO 权重路径、数据集目录、训练输出目录全部写死
``D:\\dehaze_fusion\\...``, 且推理脚本内嵌"文件哈希触发自动重训"。
本封装:
- 权重路径显式传参 (无隐藏默认);
- 检测与训练彻底分离, 训练由 icewave.detect.yolo:train 显式调用;
- ultralytics 未安装时给出清晰指引而非 ImportError 裸异常。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

YOLO_CLASSES = ["insulator", "power_line", "ice", "tower"]
YOLO_COLORS = {
    "insulator": (0, 165, 255),
    "power_line": (255, 0, 0),
    "ice": (0, 0, 255),
    "tower": (0, 255, 0),
}


def _import_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as e:
        raise ImportError(
            "检测功能需要 ultralytics: pip install 'ultralytics>=8.2'\n"
            "注意: ultralytics YOLOv8 采用 AGPL-3.0 许可, 商业使用需评估"
            "(详见 NOTICE.md)。"
        ) from e


class YOLODetector:
    """加载 YOLOv8 权重并做与旧管线一致的检测输出格式."""

    def __init__(self, weights: str | Path, conf: float = 0.25):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"YOLO 权重不存在: {self.weights}\n"
                f"请用 scripts/download_weights.py 下载, 或显式传入 --detector-weights"
            )
        YOLO = _import_yolo()
        self.model = YOLO(str(self.weights))
        self.conf = conf

    def detect(self, img_bgr, conf: Optional[float] = None) -> list[dict]:
        results = self.model(img_bgr, conf=conf or self.conf, verbose=False)
        detections = []
        names = results[0].names if results else YOLO_CLASSES
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "class": names[cls_id] if cls_id < len(names) else str(cls_id),
                    "cls_id": cls_id,
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(box.conf[0]),
                })
        return detections

    @staticmethod
    def draw(img_bgr, detections):
        import cv2

        annotated = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = YOLO_COLORS.get(det["class"], (128, 128, 128))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated


def filter_ice_detections(detections: list[dict]) -> list[dict]:
    """仅保留与 power_line/insulator 重叠的 ice 检测 (抑制误检)."""
    line_boxes = [d for d in detections
                  if d["class"] in ("power_line", "insulator")]
    if not line_boxes:
        return [d for d in detections if d["class"] != "ice"]

    def overlaps(det, boxes):
        dx1, dy1, dx2, dy2 = det["bbox"]
        for b in boxes:
            bx1, by1, bx2, by2 = b["bbox"]
            if max(dx1, bx1) < min(dx2, bx2) and max(dy1, by1) < min(dy2, by2):
                return True
        return False

    out = []
    for det in detections:
        if det["class"] == "ice":
            if overlaps(det, line_boxes):
                out.append(det)
        else:
            out.append(det)
    return out


def train_yolo(data_yaml: str | Path, epochs: int = 100, imgsz: int = 640,
               project: str | Path = "outputs/yolo", name: str = "power_line",
               device: Optional[int] = 0):
    """显式训练入口 (替代旧版推理期自动重训)."""
    YOLO = _import_yolo()
    model = YOLO("yolov8n.pt")
    return model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                       project=str(project), name=name, device=device)
