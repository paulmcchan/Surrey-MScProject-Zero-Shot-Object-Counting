# -*- coding: utf-8 -*-
"""
Dataset loaders for the MSc Zero-Shot Object Counting project.

Provides:
- FSC147Dataset
- OmniCount191Dataset
- collate_keep_dicts
"""
# ============================================================
# FSC-147 Dataset Loader
# ============================================================

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

class FSC147Dataset(Dataset):
    """
    FSC-147 loader using the common dataset schema.

    Expected FSC-147 structure:

        FSC147_RAW/
        ├── images_384_VarV2/
        ├── gt_density_map_adaptive_384_VarV2/
        ├── annotation_FSC147_384.json
        ├── Train_Test_Val_FSC_147.json
        └── ImageClasses_FSC147.txt

    Ground truth:
        - Complete object annotations are POINT annotations.
        - box_examples_coordinates contains exemplar boxes only.
        - Exemplar boxes are NOT complete per-instance GT boxes.
    """

    def __init__(self, root, split="train", load_images=True):

        # ----------------------------------------------------
        # Validate split
        # ----------------------------------------------------

        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Invalid split '{split}'. "
                "Expected 'train', 'val', or 'test'."
            )

        self.root = Path(root)
        self.split = split
        self.load_images = load_images

        # ----------------------------------------------------
        # FSC-147 paths
        # ----------------------------------------------------

        self.image_dir = self.root / "images_384_VarV2"
        self.annotation_file = self.root / "annotation_FSC147_384.json"
        self.split_file = self.root / "Train_Test_Val_FSC_147.json"
        self.classes_file = self.root / "ImageClasses_FSC147.txt"

        # ----------------------------------------------------
        # Load annotations
        # ----------------------------------------------------

        with open(self.annotation_file, "r") as f:
            self.annotations = json.load(f)

        # ----------------------------------------------------
        # Load train / val / test split
        # ----------------------------------------------------

        with open(self.split_file, "r") as f:
            split_data = json.load(f)

        self.image_ids = split_data[self.split]

        # ----------------------------------------------------
        # Load image -> class mapping
        # ----------------------------------------------------

        self.image_to_class = {}

        with open(self.classes_file, "r") as f:
            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) >= 2:
                    image_id = parts[0]
                    class_name = parts[1]

                    self.image_to_class[image_id] = class_name


    def __len__(self):

        return len(self.image_ids)


    def __getitem__(self, idx):

        # ----------------------------------------------------
        # Basic sample information
        # ----------------------------------------------------

        image_id = self.image_ids[idx]
        image_path = self.image_dir / image_id

        ann = self.annotations[image_id]

        class_name = self.image_to_class.get(
            image_id,
            "unknown"
        )

        # ----------------------------------------------------
        # Ground-truth points
        # ----------------------------------------------------

        points = ann.get("points", [])

        # ----------------------------------------------------
        # FSC-147 exemplar boxes
        #
        # These are NOT complete GT instance boxes.
        # Keep them separate from categories["boxes"].
        # ----------------------------------------------------

        exemplar_boxes_raw = ann.get(
            "box_examples_coordinates",
            []
        )

        exemplar_boxes = []

        for box in exemplar_boxes_raw:

            # FSC-147 exemplar coordinates are commonly stored
            # as four corner points:
            #
            # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            #
            # Convert them to [xmin, ymin, xmax, ymax].

            if len(box) == 4 and hasattr(box[0], "__len__"):

                xs = [p[0] for p in box]
                ys = [p[1] for p in box]

                exemplar_boxes.append([
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                ])

        # ----------------------------------------------------
        # Image dimensions
        # ----------------------------------------------------

        width = None
        height = None
        image = None

        # FSC-147 annotation metadata contains dimensions
        # indirectly through H/W and resize ratios in the
        # standard release.

        if "W" in ann and "H" in ann:

            width = int(ann["W"])
            height = int(ann["H"])

        # If the image itself is requested, load it and use
        # its actual dimensions.

        if self.load_images:

            image = Image.open(image_path).convert("RGB")
            width, height = image.size

        # ----------------------------------------------------
        # Common sample schema
        # ----------------------------------------------------

        sample = {

            "image_id": str(image_id),

            "image_path": str(image_path),

            "image": image,

            "width": width,
            "height": height,

            "categories": {

                class_name: {

                    "points": points,

                    # FSC-147 has no complete GT instance boxes
                    "boxes": [],

                    "count": len(points),
                }
            },

            "exemplar_boxes": exemplar_boxes,

            "dataset": "fsc147",

            "domain": None,

            "split": self.split,
        }

        return sample

# ============================================================
# OmniCount-191 Dataset Loader
# ============================================================

from collections import defaultdict

OMNICOUNT_DOMAINS = [
    "Birds",
    "Fruits",
    "Household",
    "Pets",
    "Satellite",
    "Supermarket",
    "Urban",
    "Vegetables",
    "Wild",
]

class OmniCount191Dataset(Dataset):
    """
    OmniCount-191 loader using the common dataset schema.

    Actual downloaded structure:

        OMNICOUNT_ROOT/
        ├── Birds/
        │   ├── train/
        │   ├── valid/
        │   └── test/
        ├── Fruits/
        │   ├── train/
        │   ├── valid/
        │   └── test/
        ├── Household/
        │   └── train/
        └── ...

    Each available split directory contains:

        *.jpg
        _annotations.coco.json
        <domain>_vae_<split>.json

    Ground truth used by this loader:
        - COCO instance bounding boxes
        - category/class labels

    VAE JSON and sparse segmentation fields are not required for
    the common counting loader.
    """

    def __init__(
        self,
        root,
        split="train",
        domains=None,
        load_images=True
    ):

        # ----------------------------------------------------
        # Validate common split name
        # ----------------------------------------------------

        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Invalid split '{split}'. "
                "Expected 'train', 'val', or 'test'."
            )

        self.root = Path(root)
        self.split = split
        self.load_images = load_images

        # OmniCount uses "valid" on disk,
        # while our common schema uses "val".
        self.disk_split = (
            "valid" if split == "val" else split
        )

        # ----------------------------------------------------
        # Select domains
        # ----------------------------------------------------

        if domains is None:
            self.domains = list(OMNICOUNT_DOMAINS)

        elif isinstance(domains, str):
            self.domains = [domains]

        else:
            self.domains = list(domains)

        # Validate requested domains
        unknown_domains = set(self.domains) - set(OMNICOUNT_DOMAINS)

        if unknown_domains:
            raise ValueError(
                f"Unknown OmniCount domain(s): {unknown_domains}"
            )

        # ----------------------------------------------------
        # Build dataset index
        #
        # One record = one image.
        # ----------------------------------------------------

        self.index = []

        for domain in self.domains:

            split_dir = (
                self.root
                / domain
                / self.disk_split
            )

            coco_path = (
                split_dir
                / "_annotations.coco.json"
            )

            # Some OmniCount domains do not contain
            # every train / valid / test split.
            if not coco_path.exists():
                continue

            # ------------------------------------------------
            # Load local COCO annotation file
            # ------------------------------------------------

            with open(coco_path, "r") as f:
                coco = json.load(f)

            # -----------------------------------------------
            # Local category lookup
            # -----------------------------------------------

            categories_by_id = {
                cat["id"]: cat["name"]
                for cat in coco["categories"]
            }

            # -----------------------------------------------
            # Group object annotations by local image_id
            # -----------------------------------------------

            annotations_by_image = defaultdict(list)

            for ann in coco["annotations"]:
                annotations_by_image[
                    ann["image_id"]
                ].append(ann)

            # -----------------------------------------------
            # Add each image to global loader index
            # -----------------------------------------------

            for img_info in coco["images"]:

                local_image_id = img_info["id"]

                self.index.append({
                    "domain": domain,
                    "split_dir": split_dir,
                    "image_info": img_info,
                    "annotations": annotations_by_image.get(
                        local_image_id,
                        []
                    ),
                    "categories_by_id": categories_by_id,
                })


    def __len__(self):

        return len(self.index)


    def __getitem__(self, idx):

        record = self.index[idx]

        domain = record["domain"]
        split_dir = record["split_dir"]

        img_info = record["image_info"]
        annotations = record["annotations"]
        categories_by_id = record["categories_by_id"]

        # ----------------------------------------------------
        # Local COCO image information
        # ----------------------------------------------------

        local_image_id = img_info["id"]
        file_name = img_info["file_name"]

        width = int(img_info["width"])
        height = int(img_info["height"])

        image_path = split_dir / file_name

        # Globally unambiguous identifier for common schema
        image_id = (
            f"{domain}/"
            f"{self.split}/"
            f"{local_image_id}"
        )

        # ----------------------------------------------------
        # Group GT boxes by semantic class
        # ----------------------------------------------------

        boxes_by_class = defaultdict(list)

        for ann in annotations:

            category_id = ann["category_id"]

            class_name = categories_by_id[
                category_id
            ]

            # COCO format:
            # [x, y, width, height]
            x, y, w, h = ann["bbox"]

            # Convert to common:
            # [x1, y1, x2, y2]
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)

            # Clip to image boundaries.
            #
            # Our inspection found a very small number
            # of sub-pixel boundary exceedances.
            x1 = max(0.0, min(x1, float(width)))
            y1 = max(0.0, min(y1, float(height)))
            x2 = max(0.0, min(x2, float(width)))
            y2 = max(0.0, min(y2, float(height)))

            boxes_by_class[class_name].append([
                x1,
                y1,
                x2,
                y2
            ])

        # ----------------------------------------------------
        # Build common category structure
        # ----------------------------------------------------

        categories = {}

        for class_name, boxes in boxes_by_class.items():

            categories[class_name] = {
                "points": [],
                "boxes": boxes,
                "count": len(boxes),
            }

        # ----------------------------------------------------
        # Load image only if requested
        # ----------------------------------------------------

        image = None

        if self.load_images:

            image = Image.open(
                image_path
            ).convert("RGB")

            # Verify/use actual dimensions
            width, height = image.size

        # ----------------------------------------------------
        # Common sample schema
        # ----------------------------------------------------

        sample = {

            "image_id": image_id,

            "image_path": str(image_path),

            "image": image,

            "width": width,
            "height": height,

            "categories": categories,

            # OmniCount does not use FSC-style
            # exemplar boxes.
            "exemplar_boxes": [],

            "dataset": "omnicount191",

            "domain": domain,

            # Common external split naming:
            # train / val / test
            "split": self.split,
        }

        return sample

# ============================================================
# Common PyTorch DataLoader
# ============================================================

from typing import List

def collate_keep_dicts(batch: List[dict]) -> List[dict]:
    """
    Keep each sample as an individual dictionary.

    Counting samples contain:
        - variable numbers of classes
        - variable numbers of GT points
        - variable numbers of GT boxes

    Therefore, the default PyTorch collate function should not
    attempt to stack these ragged structures into tensors.

    Tensor conversion should be performed later by whichever
    pipeline stage requires it.
    """
    return batch