"""CPU tests for the B6 loaders (CARPKDataset, VOC2007CountingDataset) on synthetic data.

Checks: common schema; COCO box conversion and clipping; counts; VOC 1-based box conversion;
the 'difficult' convention (count / count_all / count_nondifficult); prompt names; 20-class vocabulary.
"""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import CARPKDataset, VOC2007CountingDataset, VOC_CLASSES, voc_prompt_name


def make_carpk(d):
    base = Path(d) / "carpk_coco"
    for split, n in [("test", 3), ("train", 2)]:
        (base / split).mkdir(parents=True)
        images = [{"id": i + 1, "file_name": f"2016_TPZ_{i:05d}.png", "width": 1280, "height": 720} for i in range(n)]
        anns, k = [], 0
        for im in images:
            for j in range(im["id"] * 2):                      # 2, 4, 6 cars
                k += 1
                anns.append({"id": k, "image_id": im["id"], "category_id": 1, "bbox": [10 * j, 5, 20, 10],
                             "area": 200, "iscrowd": 0})
        anns.append({"id": k + 1, "image_id": 1, "category_id": 1, "bbox": [1270, 715, 30, 30], "area": 900, "iscrowd": 0})
        (base / f"{split}_annotations.json").write_text(json.dumps(
            {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "car"}]}))
    return Path(d)


def make_voc(d):
    voc = Path(d) / "VOCdevkit" / "VOC2007"
    for sub in ["Annotations", "ImageSets/Main", "JPEGImages"]:
        (voc / sub).mkdir(parents=True)
    objs = {"000001": [("dog", 0), ("dog", 1), ("person", 0)],
            "000002": [("pottedplant", 1)],
            "000004": [("person", 0), ("person", 0), ("tvmonitor", 0)]}
    for iid, lst in objs.items():
        xml = "<annotation><size><width>500</width><height>375</height><depth>3</depth></size>"
        for name, diff in lst:
            xml += (f"<object><name>{name}</name><difficult>{diff}</difficult>"
                    "<bndbox><xmin>11</xmin><ymin>21</ymin><xmax>100</xmax><ymax>200</ymax></bndbox></object>")
        (voc / "Annotations" / f"{iid}.xml").write_text(xml + "</annotation>")
    (voc / "ImageSets/Main/test.txt").write_text("\n".join(objs) + "\n")
    return Path(d)


def test_carpk():
    with tempfile.TemporaryDirectory() as d:
        root = make_carpk(d)
        ds = CARPKDataset(root, "test", load_images=False)
        assert len(ds) == 3
        counts = [ds[i]["categories"]["car"]["count"] for i in range(3)]
        assert counts == [3, 4, 6], counts                     # image 1: 2 cars + 1 boundary box
        s = ds[0]
        assert s["dataset"] == "carpk" and s["image_id"].startswith("CARPK/test/") and s["image_path"].endswith(".png")
        b = s["categories"]["car"]["boxes"]
        assert b[0] == [0.0, 5.0, 20.0, 15.0]                  # x,y,w,h -> x1,y1,x2,y2
        assert b[-1] == [1270.0, 715.0, 1280.0, 720.0]         # clipped to the image
        assert len(CARPKDataset(root, "train", load_images=False)) == 2


def test_voc_difficult_convention():
    with tempfile.TemporaryDirectory() as d:
        root = make_voc(d)
        all_ = VOC2007CountingDataset(root, include_difficult=True, load_images=False)
        nd = VOC2007CountingDataset(root, include_difficult=False, load_images=False)
        assert len(all_) == 3
        a0, n0 = all_[0]["categories"], nd[0]["categories"]
        assert a0["dog"]["count"] == 2 and n0["dog"]["count"] == 1
        assert a0["dog"]["count_all"] == 2 and a0["dog"]["count_nondifficult"] == 1
        assert "pottedplant" in all_[1]["categories"] and "pottedplant" not in nd[1]["categories"]
        assert all_[2]["categories"]["person"]["count"] == 2
        assert a0["dog"]["boxes"][0] == [10.0, 20.0, 100.0, 200.0]   # 1-based -> 0-based
        s = all_[0]
        assert s["domain_vocabulary"] == VOC_CLASSES and len(VOC_CLASSES) == 20
        assert s["prompt_names"]["diningtable"] == "dining table" and voc_prompt_name("cat") == "cat"
        assert s["image_path"].endswith("JPEGImages/000001.jpg") and s["dataset"] == "voc2007"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
