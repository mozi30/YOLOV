import os
import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess, vis

#Visdrone classes
CLASS_NAMES = (
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
)

def draw_gt(img, anns, coco, color=(0, 255, 0)):
    h, w = img.shape[:2]
    for ann in anns:
        x, y, bw, bh = ann["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + bw), int(y + bh)
        cat_id = ann["category_id"]
        cls_name = coco.cats[cat_id]["name"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"GT:{cls_name}", (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def main():
    # adjust paths / exp / ckpt if needed
    exp_file = "/home/mozi/models/yolox-swintiny/exp.py"
    ckpt_file = "/home/mozi/models/yolox-swintiny/best_ckpt.pth"
    out_dir = "test_eval_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # build exp, model, dataloader same as eval
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.eval().cuda()

    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded ckpt. Missing:", len(missing), "Unexpected:", len(unexpected))

    val_loader = exp.get_eval_loader(batch_size=1, is_distributed=False)

    coco = val_loader.dataset.coco
    preproc = ValTransform()
    test_size = exp.test_size
    num_images = 10  # how many to visualize

    with torch.no_grad():
        for i, (imgs, _, info_imgs, ids) in enumerate(val_loader):
            if i >= num_images:
                break

            img_id = int(ids[0].item() if torch.is_tensor(ids[0]) else ids[0])
            img_info = coco.loadImgs([img_id])[0]
            file_name = img_info["file_name"]
            img_path = os.path.join(exp.data_dir, file_name)
            print("Processing:", img_path)

            raw_img = cv2.imread(img_path)
            if raw_img is None:
                print("  -> cannot read image, skip")
                continue

            # run model on this one image (same resize as eval)
            h, w = raw_img.shape[:2]
            ratio = min(test_size[0] / h, test_size[1] / w)
            img_in, _ = preproc(raw_img, None, test_size)
            img_in = torch.from_numpy(img_in).unsqueeze(0).float().cuda()

            outputs = model(img_in)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

            # draw predictions (red)
            pred_img = raw_img.copy()
            if outputs[0] is not None:
                out = outputs[0].cpu()
                bboxes = out[:, :4] / ratio
                cls = out[:, 6]
                scores = out[:, 4] * out[:, 5]
                pred_img = vis(pred_img, bboxes, scores, cls,
                               exp.test_conf, CLASS_NAMES, t_size=1)

            # draw GT (green)
            gt_img = raw_img.copy()
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)
            gt_img = draw_gt(gt_img, anns, coco)

            # concat: [GT | Pred]
            gt_resized = cv2.resize(gt_img, (raw_img.shape[1], raw_img.shape[0]))
            pred_resized = cv2.resize(pred_img, (raw_img.shape[1], raw_img.shape[0]))
            concat = cv2.hconcat([gt_resized, pred_resized])

            save_name = file_name.replace("/", "_")
            out_path = os.path.join(out_dir, save_name)
            print("Saving:", out_path)
            cv2.imwrite(out_path, concat)

if __name__ == "__main__":
    main()