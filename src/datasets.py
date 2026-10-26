# -*- coding: utf-8 -*-
"""
Dataset loaders for the MSc Zero-Shot Object Counting project.

Provides:
- FSC147Dataset
- OmniCount191Dataset
- CARPKDataset (B6)
- VOC2007CountingDataset (B6)
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

# ------------------------------------------------------------
# Roboflow-export placeholder classes (category id 0 used as a
# supercategory label). They are excluded from each domain's
# query vocabulary. B3R Section 5 audits whether any of them
# actually carries annotations.
# ------------------------------------------------------------
OMNICOUNT_PLACEHOLDER_CLASSES = {
    "autonomousvehicle",   # Urban
    "lands",               # Satellite
    "Fruits",              # Vegetables
}

# ------------------------------------------------------------
# Text-prompt names (used ONLY to prompt the VLM).
# GT class keys are never changed; evaluation always uses the
# original annotation names. Any class not listed here is
# prompted as: lower-case, underscores -> spaces.
# ------------------------------------------------------------
OMNICOUNT_PROMPT_NAMES = {
    # typos / joined words
    "oren": "orange",
    "raddish": "radish",
    "Cerealbox": "cereal box",
    "trafficlight": "traffic light",
    "trafficlight_green": "green traffic light",
    "trafficlight_red": "red traffic light",
    "trafficlight_yellow": "yellow traffic light",
    # product / jargon names
    "planet-oat-original": "oat milk carton",
    "Galia": "galia melon",
    # word order
    "Orange green": "green orange",
    "Orange ripe": "ripe orange",
    # Urban: disambiguate bicycle vs motorbike riders
    "bike": "bicycle",
    "biker": "cyclist",
    "motorbiker": "motorcyclist",
}


def omnicount_prompt_name(class_name):
    """Annotation class name -> text prompt for the VLM."""
    if class_name in OMNICOUNT_PROMPT_NAMES:
        return OMNICOUNT_PROMPT_NAMES[class_name]
    return class_name.replace("_", " ").lower()

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

        # Per-domain query vocabulary for this split:
        # all listed category names except placeholders.
        self.domain_vocabulary = {}

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

            self.domain_vocabulary[domain] = sorted(
                set(categories_by_id.values())
                - OMNICOUNT_PLACEHOLDER_CLASSES
            )

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

            # OmniCount GT is boxes only. Box centres provide the
            # point representation required by point-based
            # diagnostics (e.g. GT-in-mask tests).
            points = [
                [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
                for b in boxes
            ]

            categories[class_name] = {
                "points": points,
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

            # Classes to query for this image (its domain's
            # vocabulary) and their text prompts.
            "domain_vocabulary": list(
                self.domain_vocabulary[domain]
            ),
            "prompt_names": {
                c: omnicount_prompt_name(c)
                for c in self.domain_vocabulary[domain]
            },

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

# ============================================================
# CARPK Dataset Loader (B6)
# ============================================================

class CARPKDataset(Dataset):
    """
    CARPK (Hsieh et al., ICCV 2017) in COCO format.

    Expected structure (as in the verified Kaggle conversion 'carpk_coco'):

        root/ (anywhere below)
        ├── test/                     459 images (1280 x 720)
        ├── train/                    989 images
        ├── test_annotations.json     COCO: images, annotations (bbox x,y,w,h), categories [car]
        └── train_annotations.json

    Ground truth: one box per car; count = number of boxes (as given, no de-duplication).
    """

    CLASS_NAME = "car"

    def __init__(self, root, split="test", load_images=True):
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split '{split}'. Use 'train' or 'test'.")
        self.root = Path(root)
        self.split = split
        self.load_images = load_images
        found = sorted(self.root.rglob(f"{split}_annotations.json"))
        if not found:
            raise FileNotFoundError(f"{split}_annotations.json not found under {self.root}")
        self.annotation_file = found[0]
        self.image_dir = self.annotation_file.parent / split
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")
        coco = json.loads(self.annotation_file.read_text())
        self.images = sorted(coco["images"], key=lambda im: im["file_name"])
        self.boxes_by_image = defaultdict(list)
        for a in coco["annotations"]:
            self.boxes_by_image[a["image_id"]].append(a["bbox"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        width, height = int(im["width"]), int(im["height"])
        image_path = self.image_dir / im["file_name"]
        boxes = []
        for x, y, w, h in self.boxes_by_image.get(im["id"], []):
            x1 = max(0.0, min(float(x), width)); y1 = max(0.0, min(float(y), height))
            x2 = max(0.0, min(float(x + w), width)); y2 = max(0.0, min(float(y + h), height))
            boxes.append([x1, y1, x2, y2])
        points = [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes]
        image = None
        if self.load_images:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
        return {
            "image_id": f"CARPK/{self.split}/{im['file_name']}",
            "image_path": str(image_path),
            "image": image,
            "width": width, "height": height,
            "categories": {self.CLASS_NAME: {"points": points, "boxes": boxes, "count": len(boxes)}},
            "exemplar_boxes": [],
            "dataset": "carpk",
            "domain": None,
            "split": self.split,
        }


# ============================================================
# PASCAL VOC 2007 Counting Dataset Loader (B6)
# ============================================================

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]

VOC_PROMPT_NAMES = {
    "aeroplane": "aeroplane", "diningtable": "dining table", "motorbike": "motorbike",
    "pottedplant": "potted plant", "tvmonitor": "tv monitor",
}


def voc_prompt_name(class_name):
    """Natural-language text prompt for a VOC class label."""
    return VOC_PROMPT_NAMES.get(class_name, class_name)


class VOC2007CountingDataset(Dataset):
    """
    PASCAL VOC 2007 for multi-label counting (count-test = the official 4,952-image test set).

    Expected structure (from VOCtest_06-Nov-2007.tar):

        root/ (anywhere below)
        └── VOCdevkit/VOC2007/
            ├── Annotations/*.xml
            ├── ImageSets/Main/test.txt
            └── JPEGImages/*.jpg

    Ground truth per class: instance boxes from the XML annotations.
    include_difficult decides whether objects flagged 'difficult' are counted (the B6 rule
    file fixes the convention); both counts are always provided:
        categories[c]["count"]               — under the chosen convention
        categories[c]["count_all"]           — all objects
        categories[c]["count_nondifficult"]  — difficult objects excluded
    Every image is queried for all 20 classes ("domain_vocabulary").
    """

    def __init__(self, root, split="test", include_difficult=True, load_images=True):
        self.root = Path(root)
        self.split = split
        self.include_difficult = bool(include_difficult)
        self.load_images = load_images
        found = sorted(self.root.rglob("VOC2007/ImageSets/Main"))
        if not found:
            raise FileNotFoundError(f"VOCdevkit/VOC2007 not found under {self.root}")
        self.voc_dir = found[0].parent.parent                 # .../VOC2007
        id_file = self.voc_dir / "ImageSets" / "Main" / f"{split}.txt"
        self.image_ids = id_file.read_text().split()

    def __len__(self):
        return len(self.image_ids)

    def parse(self, image_id):
        import xml.etree.ElementTree as ET
        root = ET.parse(self.voc_dir / "Annotations" / f"{image_id}.xml").getroot()
        size = root.find("size")
        width, height = int(size.findtext("width")), int(size.findtext("height"))
        objs = []
        for o in root.findall("object"):
            bb = o.find("bndbox")
            # VOC boxes are 1-based inclusive pixel coordinates -> 0-based [x1, y1, x2, y2]
            box = [float(bb.findtext("xmin")) - 1, float(bb.findtext("ymin")) - 1,
                   float(bb.findtext("xmax")), float(bb.findtext("ymax"))]
            objs.append({"name": o.findtext("name").strip(), "difficult": int(o.findtext("difficult", "0") or 0),
                         "box": box})
        return width, height, objs

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        width, height, objs = self.parse(image_id)
        categories = {}
        for c in VOC_CLASSES:
            mine = [o for o in objs if o["name"] == c]
            if not mine:
                continue
            kept = [o for o in mine if self.include_difficult or not o["difficult"]]
            boxes = [o["box"] for o in kept]
            n_nd = sum(1 for o in mine if not o["difficult"])
            if not boxes and n_nd == 0 and not self.include_difficult:
                continue                                          # only difficult objects -> absent
            categories[c] = {"points": [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes],
                             "boxes": boxes, "count": len(boxes),
                             "count_all": len(mine), "count_nondifficult": n_nd}
        image_path = self.voc_dir / "JPEGImages" / f"{image_id}.jpg"
        image = None
        if self.load_images:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
        return {
            "image_id": f"VOC2007/{self.split}/{image_id}",
            "image_path": str(image_path),
            "image": image,
            "width": width, "height": height,
            "categories": categories,
            "exemplar_boxes": [],
            "dataset": "voc2007",
            "domain": None,
            "domain_vocabulary": list(VOC_CLASSES),
            "prompt_names": {c: voc_prompt_name(c) for c in VOC_CLASSES},
            "split": self.split,
        }
