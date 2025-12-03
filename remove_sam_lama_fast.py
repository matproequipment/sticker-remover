# remove_sticker_yolo_sam_fast.py
# YOLO (box) -> SAM (precise mask, now on downscaled copy) -> upsample -> LaMa inpaint (combined mask + ROI crop)
# pip install simple-lama-inpainting

import os, sys, cv2, numpy as np, torch, shutil
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from simple_lama_inpainting import SimpleLama

# ========= Config (serverless) =========
YOLO_MODEL_PATH  = "/app/best.pt"               # resolves to /app/best.pt in your container
SAM_CHECKPOINT   = "/app/sam_vit_b_01ec64.pth"  # resolves to /app/sam_vit_b_01ec64.pth

CONF             = 0.03
IMGSZ            = 896
INPAINT_METHOD   = "lama"
TELEA_RADIUS     = 2
NS_RADIUS        = 2
SMALL_MASK_AREA  = 1600
MAX_ROI_PIXELS   = 1_500_000
ROI_PAD_PX       = 25

ERODE_PX         = 0
DILATE_PX        = 2
FEATHER_PX       = 3

SAM_MAX_SIDE     = 1600

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= Load models once at import time (perfect for serverless) =========
print(f"Loading YOLO from {YOLO_MODEL_PATH}...")
yolo = YOLO(YOLO_MODEL_PATH)
yolo.fuse()
yolo.to(DEVICE)
yolo.model.eval()

# Optional ultra-speed boost
if hasattr(torch, "compile"):
    print("Compiling YOLO model for maximum speed...")
    yolo.model = torch.compile(yolo.model, mode="max-autotune", fullgraph=True)

print(f"Loading SAM (vit_b) from {SAM_CHECKPOINT}...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.eval()
sam.to(DEVICE)
predictor = SamPredictor(sam)

print("Loading LaMA inpainter...")
lama_model = SimpleLama(device=DEVICE)

# Legacy batch-mode directories (safe defaults, never used in serverless)
SAVE_DEBUG_MASKS = False
DEBUG_DIR = "./_debug_masks"
OUTPUT_DIR = "./cover_thumb/output"
INPUT_DIR = "./cover_thumb/input"
NOTFOUND_DIR = "./cover_thumb/not-found"

print("All models loaded successfully!")

def _copy_to_notfound(src_path: str):
    """
    Serverless version: do nothing.
    We already return the original image in the two main cases above.
    This function exists only to prevent NameError.
    """
    pass

def _tight_feather(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    if ERODE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX*2+1, ERODE_PX*2+1))
        m = cv2.erode(m, k, 1)
    if DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX*2+1, DILATE_PX*2+1))
        m = cv2.dilate(m, k, 1)
    if FEATHER_PX > 0:
        m = cv2.GaussianBlur(m, (FEATHER_PX*2+1, FEATHER_PX*2+1), 0)
    return m

def _seamless_local(img: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    nz = cv2.findNonZero((mask > 0).astype(np.uint8))
    if nz is None:
        return None
    x, y, w, h = cv2.boundingRect(nz)
    x, y, w, h = int(x), int(y), int(w), int(h)
    x0, y0 = max(0, x-8), max(0, y-8)
    x1, y1 = min(img.shape[1], x+w+8), min(img.shape[0], y+h+8)

    roi  = img[y0:y1, x0:x1].copy()
    src  = cv2.GaussianBlur(roi, (31,31), 0)
    mloc = np.zeros_like(roi[:,:,0])
    mloc[mask[y0:y1, x0:x1] > 0] = 255
    center = ((x + w//2) - x0, (y + h//2) - y0)

    try:
        blended = cv2.seamlessClone(src, roi, mloc, center, cv2.MIXED_CLONE)
        out = img.copy()
        out[y0:y1, x0:x1] = blended
        return out
    except Exception:
        return None

def telea_inpaint(img_bgr: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    mask = _tight_feather(raw_mask)
    return cv2.inpaint(img_bgr, mask, TELEA_RADIUS, cv2.INPAINT_TELEA)

def ns_inpaint(img_bgr: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    mask = _tight_feather(raw_mask)
    return cv2.inpaint(img_bgr, mask, NS_RADIUS, cv2.INPAINT_NS)

def lama_inpaint(img_bgr: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    mask = _tight_feather(raw_mask)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        out = lama_model(img_rgb, mask)
    if hasattr(out, "mode"):
        out = np.array(out)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def smart_inpaint(img: np.ndarray, raw_mask: np.ndarray, method: str = "lama") -> np.ndarray:
    if method == "lama":
        return lama_inpaint(img, raw_mask)
    if method == "telea":
        return telea_inpaint(img, raw_mask)
    if method == "ns":
        return ns_inpaint(img, raw_mask)
    if method in ("seamless", "auto"):
        mask = _tight_feather(raw_mask)
        out = _seamless_local(img, mask)
        if out is not None:
            return out
        try:
            return lama_inpaint(img, mask)
        except Exception:
            return telea_inpaint(img, mask)
    return ns_inpaint(img, raw_mask)

def _resize_for_sam(img_bgr: np.ndarray, max_side: int = SAM_MAX_SIDE):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    scale = min(max_side / max(H, W), 1.0)
    if scale < 1.0:
        newW, newH = int(W * scale), int(H * scale)
        small = cv2.resize(img_rgb, (newW, newH), interpolation=cv2.INTER_AREA)
        return small, scale
    return img_rgb, 1.0

def _upsample_mask_bool(mask_small: np.ndarray, orig_shape_hw: tuple[int,int]) -> np.ndarray:
    H, W = orig_shape_hw
    m_u8 = (mask_small.astype(np.uint8) * 255)
    big = cv2.resize(m_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    return (big > 0)

def sam_prepare_image_once(img_bgr: np.ndarray):
    small_rgb, sam_scale = _resize_for_sam(img_bgr, SAM_MAX_SIDE)
    predictor.set_image(small_rgb)
    return sam_scale, img_bgr.shape[:2]

def sam_mask_from_box_scaled(box_xyxy: list[float], sam_scale: float, orig_hw: tuple[int,int]) -> np.ndarray:
    b = np.array(box_xyxy, dtype=np.float32) * float(sam_scale)
    H, W = orig_hw
    with torch.inference_mode():
        dev = predictor.device
        b_t = torch.as_tensor(b, dtype=torch.float32, device=dev).unsqueeze(0)
        b_trans = predictor.transform.apply_boxes_torch(b_t, (int(H * sam_scale), int(W * sam_scale)))
        try:
            masks_t, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=b_trans,
                multimask_output=False
            )
            mask_small = masks_t[0, 0].to("cpu").numpy().astype(bool)
        except TypeError:
            b_np = np.array([b], dtype=np.float32)
            b_trans_np = predictor.transform.apply_boxes(b_np, (int(H * sam_scale), int(W * sam_scale)))
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=b_trans_np[0],
                multimask_output=False
            )
            mask_small = masks[0].astype(bool)

    mask_big_bool = _upsample_mask_bool(mask_small, (H, W))
    return (mask_big_bool.astype(np.uint8)) * 255

def yolo_box_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int, pad: int = 2) -> np.ndarray:
    bx1 = max(0, x1 - pad); by1 = max(0, y1 - pad)
    bx2 = min(w - 1, x2 + pad); by2 = min(h - 1, y2 + pad)
    m = np.zeros((h, w), np.uint8)
    m[by1:by2, bx1:bx2] = 255
    return m

def combine_masks(masks: list[np.ndarray], h: int, w: int) -> np.ndarray:
    if not masks:
        return np.zeros((h, w), np.uint8)
    m = np.zeros((h, w), np.uint8)
    for mk in masks:
        if mk is None:
            continue
        if mk.ndim == 3:
            mk = cv2.cvtColor(mk, cv2.COLOR_BGR2GRAY)
        m = cv2.bitwise_or(m, (mk > 0).astype(np.uint8) * 255)
    return m

def bbox_from_mask(mask: np.ndarray, pad: int = ROI_PAD_PX, shape: tuple[int,int] | None = None):
    nz = cv2.findNonZero((mask > 0).astype(np.uint8))
    if nz is None:
        return None
    x, y, w, h = cv2.boundingRect(nz)
    H, W = (mask.shape[0], mask.shape[1]) if shape is None else shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return x0, y0, x1, y1

def inpaint_on_roi(img: np.ndarray, full_mask: np.ndarray, method: str = "lama",
                   max_roi_pixels: int = MAX_ROI_PIXELS) -> np.ndarray:
    total_area = int(np.count_nonzero(full_mask))
    if total_area <= SMALL_MASK_AREA:
        return telea_inpaint(img, full_mask)

    box = bbox_from_mask(full_mask, pad=ROI_PAD_PX, shape=img.shape[:2])
    if box is None:
        return img
    x0, y0, x1, y1 = box

    roi_img  = img[y0:y1, x0:x1].copy()
    roi_mask = full_mask[y0:y1, x0:x1].copy()

    target_h = y1 - y0
    target_w = x1 - x0

    H, W = roi_img.shape[:2]
    pixels = H * W
    if pixels > max_roi_pixels:
        scale = (max_roi_pixels / float(pixels)) ** 0.5
        newW = max(64, int(round(W * scale)))
        newH = max(64, int(round(H * scale)))
        roi_img  = cv2.resize(roi_img, (newW, newH), interpolation=cv2.INTER_AREA)
        roi_mask = cv2.resize(roi_mask, (newW, newH), interpolation=cv2.INTER_NEAREST)

    cleaned = smart_inpaint(roi_img, roi_mask, method=method)

    if cleaned.shape[0] != target_h or cleaned.shape[1] != target_w:
        cleaned = cv2.resize(cleaned, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        cleaned = cleaned[:target_h, :target_w]

    out = img.copy()
    out[y0:y1, x0:x1] = cleaned
    return out

def process_image(path_in: str, path_out: str, slno: int):
    img = cv2.imread(path_in)
    if img is None:
        print("Skip (unreadable):", path_in); return

    with torch.inference_mode():
        res = yolo(img, conf=CONF, iou=0.4, imgsz=IMGSZ, augment=False, verbose=False, device=DEVICE)
    boxes = res[0].boxes.xyxy.detach().to("cpu").numpy()

    if boxes.shape[0] == 0:
        print(slno, "- No detection:", os.path.basename(path_in))
        shutil.copy2(path_in, path_out)
        return

    h, w = img.shape[:2]
    masks_for_all = []

    sam_scale, orig_hw = sam_prepare_image_once(img)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        xi1, yi1, xi2, yi2 = map(int, [x1, y1, x2, y2])
        box_m = yolo_box_mask(h, w, xi1, yi1, xi2, yi2, pad=2)

        sam_m = sam_mask_from_box_scaled([x1, y1, x2, y2], sam_scale, orig_hw)
        sam_m = cv2.bitwise_and(sam_m, box_m)

        if SAVE_DEBUG_MASKS:
            overlay = img.copy()
            overlay[sam_m > 0] = (0.6 * overlay[sam_m > 0] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{os.path.basename(path_in)}_mask{i}.png"), overlay)

        masks_for_all.append(sam_m if int(np.count_nonzero(sam_m)) >= 50 else box_m)

    full_mask = combine_masks(masks_for_all, h, w)
    if int(np.count_nonzero(full_mask)) == 0:
        print("Masks empty after combine:", os.path.basename(path_in))
        _copy_to_notfound(path_in)
        return

    img = inpaint_on_roi(img, full_mask, method=INPAINT_METHOD)

    try:
        cv2.imwrite(path_out, img, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
    except Exception:
        cv2.imwrite(path_out, img)

    print(f"{slno} - Cleaned fast ({INPAINT_METHOD.upper()}+ROI+SAM@{SAM_MAX_SIDE}): {os.path.basename(path_in)}")

def process_image_full(path_in: str, path_out: str, slno: int):
    """
    Variant 2: Full-image inpaint instead of ROI-only.
    Uses the same YOLO+SAM pipeline, but passes the combined mask
    for the entire image directly to the inpaint method.
    """
    img = cv2.imread(path_in)
    if img is None:
        print("Skip (unreadable):", path_in)
        return

    # YOLO inference (same as in process_image)
    with torch.inference_mode():
        res = yolo(
            img,
            conf=CONF,
            iou=0.4,
            imgsz=IMGSZ,
            augment=False,
            verbose=False,
            device=DEVICE,
        )
    boxes = res[0].boxes.xyxy.detach().to("cpu").numpy()

    if boxes.shape[0] == 0:
        # No detection → return original image unchanged
        print(slno, "- No detection (full):", os.path.basename(path_in))
        shutil.copy2(path_in, path_out)
        return

    h, w = img.shape[:2]
    masks_for_all = []

    # Prepare SAM once
    sam_scale, orig_hw = sam_prepare_image_once(img)

    # Build per-box masks
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        xi1, yi1, xi2, yi2 = map(int, [x1, y1, x2, y2])
        box_m = yolo_box_mask(h, w, xi1, yi1, xi2, yi2, pad=2)

        sam_m = sam_mask_from_box_scaled([x1, y1, x2, y2], sam_scale, orig_hw)
        sam_m = cv2.bitwise_and(sam_m, box_m)

        if SAVE_DEBUG_MASKS:
            overlay = img.copy()
            overlay[sam_m > 0] = (
                0.6 * overlay[sam_m > 0] + 0.4 * np.array([0, 0, 255])
            ).astype(np.uint8)
            cv2.imwrite(
                os.path.join(
                    DEBUG_DIR,
                    f"{os.path.basename(path_in)}_full_mask{i}.png",
                ),
                overlay,
            )

        masks_for_all.append(sam_m if int(np.count_nonzero(sam_m)) >= 50 else box_m)

    # Combine all masks
    full_mask = combine_masks(masks_for_all, h, w)
    if int(np.count_nonzero(full_mask)) == 0:
        print("Masks empty after combine (full):", os.path.basename(path_in))
        shutil.copy2(path_in, path_out)
        return

    # 🔥 Variant 2: FULL IMAGE INPAINT (no ROI crop)
    img_clean = smart_inpaint(img, full_mask, method=INPAINT_METHOD)

    try:
        cv2.imwrite(path_out, img_clean, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
    except Exception:
        cv2.imwrite(path_out, img_clean)

    print(
        f"{slno} - Cleaned FULL ({INPAINT_METHOD.upper()}+SAM@{SAM_MAX_SIDE}): "
        f"{os.path.basename(path_in)}"
    )
