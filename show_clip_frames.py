import matplotlib.pyplot as plt
import cv2

def show_clip_frames(dataset, clip_index):
    """
    Displays one clip from the dataset.
    - Top row: lframe (local frames)
    - Bottom row: gframe (global frames)
    """
    # Get list of image paths in the selected clip (e.g. 16 paths)
    clip_paths = dataset.res[clip_index]

    lframe = dataset.lframe
    gframe = dataset.gframe

    # Split into local + global
    local_frames = clip_paths[:lframe]
    global_frames = clip_paths[lframe:lframe + gframe]

    n_local = len(local_frames)
    n_global = len(global_frames)
    
    fig = plt.figure(figsize=(n_local * 3, 6))  # Adjust size

    # ---- Top row = lframe ----
    for i, path in enumerate(local_frames):
        img, _, _, _ = dataset.pull_item(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(2, max(n_local, n_global), i + 1)
        ax.imshow(img)
        ax.set_title(f"L{i}")
        ax.axis("off")

    # ---- Bottom row = gframe ----
    for i, path in enumerate(global_frames):
        img, _, _, _ = dataset.pull_item(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(2, max(n_local, n_global), max(n_local, n_global) + i + 1)
        ax.imshow(img)
        ax.set_title(f"G{i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------- Example usage ----------
if __name__ == "__main__":
    from yolox.data.datasets.vid  import VisDroneVID

    dataset = VisDroneVID(
        data_dir = "/home/mozi/datasets/visdrone/yolov",
        json_file="/home/mozi/datasets/visdrone/yolov/annotations/imagenet_vid_train.json",
        img_size=(640, 640),
        lframe=16,          # set according to your setup
        gframe=16,
        mode="gl",
        val=True           # or False depending on your setup
    )

    # Show the 1st clip (use different numbers to move between clips)
    show_clip_frames(dataset, clip_index=0)
