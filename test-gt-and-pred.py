import json
import os
import time

import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.data.datasets.vid_classes import VID_classes as CLASSES
from yolox.utils import postprocess, vis
import random

def load_annotations(ann_file, shuffle=True):
    with open(ann_file, "r") as f:
        data = json.load(f)
    images = list(data["images"])
    if shuffle:
        random.shuffle(images)
    imgs = {img["id"]: img for img in images}
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    return imgs, anns_by_img

def draw_gt(img, gt_anns, cls_names, color=(0, 255, 0)):
    for ann in gt_anns:
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cat_id = ann["category_id"]
        cls_name = cls_names[cat_id] if 0 <= cat_id < len(cls_names) else str(cat_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"GT:{cls_name}", (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def build_model(exp_file, ckpt_file, device="cuda"):
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.eval()
    model.to(device)
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return exp, model

def run_one(exp, model, img_path, device="cuda", conf=0.3, nms=0.5):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    test_size = exp.test_size
    ratio = min(test_size[0] / h, test_size[1] / w)
    preproc = ValTransform()
    img_in, _ = preproc(img, None, test_size)
    img_in = torch.from_numpy(img_in).unsqueeze(0).float().to(device)

    with torch.no_grad():
        t0 = time.time()
        outputs = model(img_in)
        outputs = postprocess(outputs, exp.num_classes, conf, nms, class_agnostic=True)
        print(f"Infer time: {time.time() - t0:.4f}s")
    return img, outputs[0], ratio

def main():
    # paths – adjust if needed
    exp_file = "exps/customed_example/yolox_swinbase.py"
    ckpt_file = "YOLOX_outputs/yolox_swinbase/best_ckpt.pth"
    ann_file = "/home/mozi/datasets/visdrone/yolov/annotations/imagenet_vid_val_coco.json"
    img_root = "/home/mozi/datasets/visdrone/yolov/Data/VID/"  # adjust if different
    out_dir = "output/gt_vs_pred"
    os.makedirs(out_dir, exist_ok=True)
    print("Loading annotations...")
    imgs, anns_by_img = load_annotations(ann_file)
    print("Building model...")
    exp, model = build_model(exp_file, ckpt_file, device="cuda")
    print("Starting visualization...")
    # pick N images to visualize
    img_ids = list(imgs.keys())[:20]
    print(f"Visualizing {len(img_ids)} images...")
    for img_id in img_ids:
        print(f"Processing image ID: {img_id}")
        img_info = imgs[img_id]
        file_name = img_info["file_name"]
        img_path = os.path.join(img_root, file_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}, skipping.")
            continue

        raw_img, pred, ratio = run_one(exp, model, img_path, device="cuda",
                                       conf=exp.test_conf, nms=exp.nmsthre)

        # draw predictions (red)
        if pred is not None:
            pred = pred.cpu()
            bboxes = pred[:, 0:4] / ratio
            cls = pred[:, 6]
            scores = pred[:, 4] * pred[:, 5]
            pred_img = vis(raw_img.copy(), bboxes, scores, cls,
                           exp.test_conf, CLASSES, t_size=1)
        else:
            pred_img = raw_img.copy()

        # draw GT (green)
        gt_img = raw_img.copy()
        gt_anns = anns_by_img.get(img_id, [])
        gt_img = draw_gt(gt_img, gt_anns, CLASSES)

        # concat side-by-side: [GT | Pred]
        h = raw_img.shape[0]
        gt_resized = cv2.resize(gt_img, (raw_img.shape[1], h))
        pred_resized = cv2.resize(pred_img, (raw_img.shape[1], h))
        concat = cv2.hconcat([gt_resized, pred_resized])

        out_path = os.path.join(out_dir, file_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print("Saving:", out_path)
        cv2.imwrite(out_path, concat)

if __name__ == "__main__":
    main()