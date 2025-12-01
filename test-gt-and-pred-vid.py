#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import time
import random

import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.data.datasets.vid_classes import VID_classes as CLASSES
from yolox.utils import postprocess, vis

# Dataset from your exp
from yolox.data.datasets.visdrone import VidDroneVIDataset


# --------------------------- Drawing GT from dataset ---------------------------

def draw_gt_from_dataset(img, dataset: VidDroneVIDataset, rel_path: str, cls_names, color=(0, 255, 0)):
    """
    Draw GT boxes from VidDroneVIDataset on the given image.

    Uses:
      - dataset.name_key_map[rel_path] -> (vid, frame_idx)
      - dataset.ann_map[(vid, frame_idx)] -> list of {"bbox": [x1,y1,x2,y2], "cid": int}
    Assumes ann_map bboxes are in *original image coordinates*.
    """
    if rel_path not in dataset.name_key_map:
        return img

    vid, frame_idx = dataset.name_key_map[rel_path]
    anns = dataset.ann_map.get((vid, frame_idx), [])

    for a in anns:
        x1, y1, x2, y2 = a["bbox"]
        cid = a["cid"]

        cls_name = cls_names[cid] if 0 <= cid < len(cls_names) else str(cid)

        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), color, 2)
        cv2.putText(
            img,
            f"GT:{cls_name}",
            (x1_i, max(0, y1_i - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return img


# --------------------------- Model building / inference ---------------------------

def build_model_from_exp(exp_file: str, ckpt_file: str, device="cuda"):
    """
    Build model via your Exp class and load checkpoint.
    """
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.eval()
    model.to(device)

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return exp, model


def run_one(exp, model, img, device="cuda", conf=0.7, nms=0.2):
    """
    Run inference on a numpy BGR image and return:
        raw_img, output, ratio, infer_time

    where `output` is either:
      - None, or
      - tensor [num_det, 7] with columns [x1, y1, x2, y2, obj, cls_conf, cls_id]
    """

    h, w = img.shape[:2]
    test_size = exp.test_size
    ratio = min(test_size[0] / h, test_size[1] / w)

    preproc = ValTransform()
    img_in, _ = preproc(img, None, test_size)
    img_in = torch.from_numpy(img_in).unsqueeze(0).float().to(device)

    with torch.no_grad():
        t0 = time.time()
        outputs = model(img_in)
        infer_time = time.time() - t0

        # ---- Normalize top-level structure ----
        main_out = outputs
        if isinstance(outputs, tuple) and len(outputs) > 0:
            # common pattern: (preds, extra)
            main_out = outputs[0]

        # ---- Case A: already postprocessed (list/tuple of per-image tensors/None) ----
        if isinstance(main_out, (list, tuple)):
            if len(main_out) == 0:
                output = None
            else:
                output = main_out[0]  # batch size 1
            return img, output, ratio, infer_time

        # ---- Case B: raw prediction tensor [B, N, 5+num_classes] ----
        if isinstance(main_out, torch.Tensor):
            pred_tensor = main_out

            if pred_tensor.dim() == 2:
                pred_tensor = pred_tensor.unsqueeze(0)  # [1, N, C]

            if pred_tensor.dim() != 3:
                raise RuntimeError(
                    f"Prediction tensor dim = {pred_tensor.dim()}, expected 2 or 3."
                )

            expected_c = 5 + exp.num_classes  # 4 bbox + obj + num_classes
            if pred_tensor.size(-1) != expected_c:
                raise RuntimeError(
                    f"Prediction tensor last dim = {pred_tensor.size(-1)}, "
                    f"but expected {expected_c} (5 + num_classes={exp.num_classes})."
                )

            outputs_pp = postprocess(
                pred_tensor,
                exp.num_classes,
                conf,
                nms,
                class_agnostic=True,
            )

            output = outputs_pp[0]  # batch size 1
            return img, output, ratio, infer_time

        # ---- Unknown structure ----
        raise RuntimeError(
            f"Unexpected model output type: {type(outputs)} "
            f"(main_out type: {type(main_out)})"
        )



# --------------------------- Main visualization using Exp + dataset ---------------------------

def main():
    # ----------------- adjust these paths -----------------
    exp_file = "/home/mozi/TemporalAttentionPlayground/YOLOV/exps/customed_example/yolov_swinbase.py"  # e.g. "exps/yolov/visdrone_vid_exp.py"
    ckpt_file = "/home/mozi/models/yolov_swinbase_w7_g_2_train_1500_val_500/best_ckpt.pth"

    # Output directory where we save per-video visualization videos
    out_dir = "output_vid_from_exp/gt_vs_pred"
    os.makedirs(out_dir, exist_ok=True)

    # ----------------- build model from Exp -----------------
    print("Building model from exp...")
    exp, model = build_model_from_exp(exp_file, ckpt_file, device="cuda")
    print(f"Model built. num_classes = {exp.num_classes}")

    # ----------------- build dataset from Exp config -----------------
    print("Building VidDroneVIDataset from exp config...")

    # We'll visualize on the validation split to avoid training augmentations
    dataset = VidDroneVIDataset(
        data_dir=exp.data_dir,
        split="val",                 # or "train" if you want
        img_size=exp.test_size,
        preproc=None,                # no augment here; we use raw images for drawing
        lframe=exp.lframe_val if hasattr(exp, "lframe_val") else 0,
        gframe=exp.gframe_val if hasattr(exp, "gframe_val") else exp.gframe,
        sample_mode="gl",
        max_epoch_samples=-1,
        gl_stride=1,
    )

    print(f"Dataset built. #videos = {len(dataset.videos)}")

    # ----------------- visualize a few videos -----------------
    max_videos = 3  # limit to first N videos
    video_indices = list(range(len(dataset.videos)))
    random.shuffle(video_indices)
    video_indices = video_indices[:max_videos]

    for vid_idx in video_indices:
        v_meta = dataset.videos[vid_idx]
        vid = v_meta["id"]
        vname = v_meta.get("video_name", f"video_{vid}")
        seq_paths = dataset.dataset_sequences[vid_idx]  # list of relative frame paths

        if not seq_paths:
            print(f"Video idx={vid_idx} has no frames, skipping.")
            continue

        print(f"\nProcessing video idx={vid_idx}, id={vid}, name={vname}, length={len(seq_paths)}")

        # Read first frame to get resolution
        first_rel_path = seq_paths[0]
        first_abs_path = os.path.join(dataset.data_dir, first_rel_path)
        first_img = cv2.imread(first_abs_path)
        if first_img is None:
            print(f"Failed to read first frame: {first_abs_path}, skipping video.")
            continue

        h0, w0 = first_img.shape[:2]

        # Prepare video writer: we will store [GT | PRED] concatenated
        out_video_path = os.path.join(
            out_dir,
            f"{os.path.splitext(os.path.basename(vname))[0]}_gt_pred.mp4",
        )
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_video_path = out_video_path.replace(".mp4", ".avi")
        #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 10  # adjust as you like
        out_w = w0 * 2
        out_h = h0
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            print(f"[ERROR] Could not open VideoWriter for {out_video_path}")
            continue

        for rel_path in seq_paths:
            abs_path = os.path.join(dataset.data_dir, rel_path)
            raw_img = cv2.imread(abs_path)
            if raw_img is None:
                print(f"Failed to read frame: {abs_path}, skipping.")
                continue

            # 1) Predictions
            raw_img_pred, pred, ratio, _ = run_one(
                exp,
                model,
                raw_img,
                device="cuda",
                conf=exp.test_conf,
                nms=exp.nmsthre,
            )

            if pred is not None:
                pred = pred.cpu()
                bboxes = pred[:, 0:4] / ratio
                cls = pred[:, 6]
                scores = pred[:, 4] * pred[:, 5]
                pred_img = vis(
                    raw_img_pred.copy(),
                    bboxes,
                    scores,
                    cls,
                    exp.test_conf,
                    CLASSES,
                    t_size=1,
                )
            else:
                pred_img = raw_img_pred.copy()

            # 2) Ground truth
            gt_img = raw_img.copy()
            gt_img = draw_gt_from_dataset(gt_img, dataset, rel_path, CLASSES)

            # 3) Concatenate side-by-side: [GT | Pred]
            # First resize both halves to the *original* first-frame size
            gt_resized = cv2.resize(gt_img, (w0, h0))
            pred_resized = cv2.resize(pred_img, (w0, h0))
            concat = cv2.hconcat([gt_resized, pred_resized])

            # Then make absolutely sure it's the size VideoWriter expects
            concat_resized = cv2.resize(concat, (out_w, out_h))
            writer.write(concat_resized)

        writer.release()
        print(f"Finished video {vname}, saved to {out_video_path}")


if __name__ == "__main__":
    main()
