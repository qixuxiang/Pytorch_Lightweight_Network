from typing import List, Dict, Sequence, Union, Tuple
from numbers import Number
import random

import numpy as np
from toolz import curry
from toolz.curried import get

from common import _tuple

__all__ = [
    "resize", "resized_crop", "center_crop", "drop_boundary_bboxes",
    "to_absolute_coords", "to_percent_coords", "hflip", "hflip2",
    "vflip", "vflip2", "random_sample_crop", "move"
]


def iou_1m(box, boxes):
    r"""
    Calculates one-to-many ious.

    Parameters
    ----------
    box : ``Sequences[Number]``
        A bounding box.
    boxes : ``array_like``
        Many bounding boxes.

    Returns
    -------
    ious : ``array_like``
        IoUs between the box and boxes.
    """
    xi1 = np.maximum(boxes[..., 0], box[0])
    yi1 = np.maximum(boxes[..., 1], box[1])
    xi2 = np.minimum(boxes[..., 2], box[2])
    yi2 = np.minimum(boxes[..., 3], box[3])
    xdiff = xi2 - xi1
    ydiff = yi2 - yi1
    inter_area = xdiff * ydiff
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[..., 2] - boxes[..., 0]) * \
        (boxes[..., 3] - boxes[..., 1])
    union_area = boxes_area + box_area - inter_area

    iou = inter_area / union_area
    iou[xdiff < 0] = 0
    iou[ydiff < 0] = 0
    return iou


def random_sample_crop(anns, size, min_iou, min_ar, max_ar, max_attemps=50):
    """
    Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    size : ``Sequence[int]``
        Size of the original image.
    min_iou : ``float``
        Minimal iou between the objects and the cropped image.
    min_ar : ``Number``
        Minimal aspect ratio.
    max_ar : ``Number``
        Maximum aspect ratio.
    max_attemps: ``int``
        Maximum attemps to try.
    """
    width, height = size
    bboxes = np.stack([ann['bbox'] for ann in anns])
    bboxes[:, 2:] += bboxes[:, :2]
    for _ in range(max_attemps):
        w = random.uniform(0.3 * width, width)
        h = random.uniform(0.3 * height, height)

        if h / w < min_ar or h / w > max_ar:
            continue

        l = random.uniform(0, width - w)
        t = random.uniform(0, height - h)
        r = l + w
        b = t + h

        patch = np.array([l, t, r, b])
        ious = iou_1m(patch, bboxes)
        if ious.min() < min_iou:
            continue

        centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
        mask = (l < centers[:, 0]) & (centers[:, 0] < r) & (
                t < centers[:, 1]) & (centers[:, 1] < b)

        if not mask.any():
            continue
        indices = np.nonzero(mask)[0].tolist()
        return get(indices, anns), l, t, w, h
    return None


@curry
def resized_crop(anns, left, upper, width, height, output_size, min_area_frac):
    anns = crop(anns, left, upper, width, height, min_area_frac)
    size = (width, height)
    # if drop:
    #     anns = drop_boundary_bboxes(anns, size)
    anns = resize(anns, size, output_size)
    return anns


@curry
def drop_boundary_bboxes(anns, size):
    r"""
    Drop bounding boxes whose centers are out of the image boundary.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    size : ``Sequence[int]``
        Size of the original image.
    """
    width, height = size
    new_anns = []
    for ann in anns:
        l, t, w, h = ann['bbox']
        x = (l + w) / 2.
        y = (t + h) / 2.
        if 0 <= x <= width and 0 <= y <= height:
            new_anns.append({**ann, "bbox": [l, t, w, h]})
    return new_anns


@curry
def center_crop(anns, size, output_size):
    r"""
    Crops the bounding boxes of the given PIL Image at the center.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    size : ``Sequence[int]``
        Size of the original image.
    output_size : ``Union[Number, Sequence[int]]``
        Desired output size of the crop. If size is an int instead of sequence like (w, h),
        a square crop (size, size) is made.
    """
    output_size = _tuple(output_size, 2)
    output_size = tuple(int(x) for x in output_size)
    w, h = size
    th, tw = output_size
    upper = int(round((h - th) / 2.))
    left = int(round((w - tw) / 2.))
    return crop(anns, left, upper, th, tw)


@curry
def crop(anns, left, upper, width, height, minimal_area_fraction=0.25):
    r"""
    Crop the bounding boxes of the given PIL Image.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    left: ``int``
        Left pixel coordinate.
    upper: ``int``
        Upper pixel coordinate.
    width: ``int``
        Width of the cropped image.
    height: ``int``
        Height of the cropped image.
    minimal_area_fraction : ``int``
        Minimal area fraction requirement.
    """
    new_anns = []
    for ann in anns:
        l, t, w, h = ann['bbox']
        area = w * h
        l -= left
        t -= upper
        if l + w >= 0 and l <= width and t + h >= 0 and t <= height:
            if l < 0:
                w += l
                l = 0
            if t < 0:
                h += t
                t = 0
            w = min(width - l, w)
            h = min(height - t, h)
            if w * h < area * minimal_area_fraction:
                continue
            new_anns.append({**ann, "bbox": [l, t, w, h]})
    return new_anns


@curry
def resize(anns, size, output_size):
    """
    Parameters
    ----------
    anns : List[Dict]
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    size : Sequence[int]
        Size of the original image.
    output_size : Union[Number, Sequence[int]]
        Desired output size. If size is a sequence like (w, h), the output size will be matched to this.
        If size is an int, the smaller edge of the image will be matched to this number maintaing
        the aspect ratio. i.e, if width > height, then image will be rescaled to
        (output_size * width / height, output_size)
    """
    w, h = size
    if isinstance(output_size, int):
        if (w <= h and w == output_size) or (h <= w and h == output_size):
            return anns
        if w < h:
            ow = output_size
            sw = sh = ow / w
        else:
            oh = output_size
            sw = sh = oh / h
    else:
        ow, oh = output_size
        sw = ow / w
        sh = oh / h
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] *= sw
        bbox[1] *= sh
        bbox[2] *= sw
        bbox[3] *= sh
        new_anns.append({**ann, "bbox": bbox})
    return new_anns


@curry
def to_percent_coords(anns, size):
    r"""
    Convert absolute coordinates of the bounding boxes to percent cocoordinates.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    size : ``Sequence[int]``
        Size of the original image.
    """
    w, h = size
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] /= w
        bbox[1] /= h
        bbox[2] /= w
        bbox[3] /= h
        new_anns.append({**ann, "bbox": bbox})
    return new_anns


@curry
def to_absolute_coords(anns, size):
    r"""
    Convert percent coordinates of the bounding boxes to absolute cocoordinates.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    size : ``Sequence[int]``
        Size of the original image.
    """
    w, h = size
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] *= w
        bbox[1] *= h
        bbox[2] *= w
        bbox[3] *= h
        new_anns.append({**ann, "bbox": bbox})
    return new_anns


@curry
def hflip(anns, size):
    """
    Horizontally flip the bounding boxes of the given PIL Image.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    size : ``Sequence[int]``
        Size of the original image.
    """
    w, h = size
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] = w - (bbox[0] + bbox[2])
        new_anns.append({**ann, "bbox": bbox})
    return new_anns


@curry
def hflip2(anns, size):
    """
    Horizontally flip the bounding boxes of the given PIL Image.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, r, b].
    size : ``Sequence[int]``
        Size of the original image.
    """
    w, h = size
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        l = bbox[0]
        bbox[0] = w - bbox[2]
        bbox[2] = w - l
        new_anns.append({**ann, "bbox": bbox})
    return new_anns


@curry
def vflip(anns, size):
    """
    Vertically flip the bounding boxes of the given PIL Image.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    size : ``Sequence[int]``
        Size of the original image.
    """
    w, h = size
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[1] = h - (bbox[1] + bbox[3])
        new_anns.append({**ann, "bbox": bbox})
    return new_anns


@curry
def vflip2(anns, size):
    r"""
    Vertically flip the bounding boxes of the given PIL Image.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    size : ``Sequence[int]``
        Size of the original image.
    """
    w, h = size

    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        t = bbox[1]
        bbox[1] = h - bbox[3]
        bbox[3] = h - t
        new_anns.append({**ann, "bbox": bbox})
    return new_anns


@curry
def move(anns, x, y):
    r"""
    Move the bounding boxes by x and y.

    Parameters
    ----------
    anns : ``List[Dict]``
        Sequences of annotation of objects, containing `bbox` of [l, t, w, h].
    x : ``Number``
        How many to move along the horizontal axis.
    y : ``Number``
        How many to move along the vertical axis.
    """

    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] += x
        bbox[1] += y
        new_anns.append({**ann, "bbox": bbox})
    return new_anns

