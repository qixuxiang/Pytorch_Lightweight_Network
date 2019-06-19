import random
import math
import warnings

from typing import Sequence, Tuple

from PIL import Image
import torchvision.transforms.functional as VF
from torchvision.transforms import ColorJitter

from transforms import JointTransform, Compose, ToTensor, InputTransform, RandomChoice, RandomApply, UseOriginal
from transforms.detection import functional as HF


class RandomExpand(JointTransform):
    """
    Expand the given PIL Image to random size.

    This is popularly used to train the SSD-like detectors.

    Parameters
    ----------
    ratios : ``tuple``
        Range of expand ratio.
    """

    def __init__(self, ratios=(1, 4)):
        super().__init__()
        self.ratios = ratios

    def __call__(self, img, anns):
        width, height = img.size
        ratio = random.uniform(*self.ratios)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        expand_image = Image.new(
            img.mode, (int(width * ratio), int(height * ratio)))
        expand_image.paste(img, (int(left), int(top)))

        new_anns = HF.move(anns, left, top)
        if len(new_anns) == 0:
            return img, anns
        return expand_image, new_anns

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += '(ratio={0})'.format(tuple(round(r, 4)
                                                    for r in self.ratios))
        return format_string



class RandomSampleCrop(JointTransform):
    """
    Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Parameters
    ----------
    min_ious : ``List[float]``
        Range of minimal iou between the objects and the cropped image.
    aspect_ratio_constraints : ``tuple``
        Range of cropped aspect ratio.
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.9), aspect_ratio_constraints=(0.5, 2)):
        super().__init__()
        self.min_ious = min_ious
        min_ar, max_ar = aspect_ratio_constraints
        self.min_ar = min_ar
        self.max_ar = max_ar

    def __call__(self, img, anns):
        min_iou = random.choice(self.min_ious)
        returns = HF.random_sample_crop(anns, img.size, min_iou, self.min_ar, self.max_ar)
        if returns is None:
            return img, anns
        else:
            anns, l, t, w, h = returns
            new_img = img.crop([l, t, l + w, t + h])
            new_anns = HF.crop(anns, l, t, w, h)
            if len(new_anns) == 0:
                return img, anns
            return new_img, new_anns


class RandomResizedCrop(JointTransform):
    """
    Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Parameters
    ----------
    size : ``Union[Number, Sequence[int]]``
        Desired output size of the crop. If size is an int instead of sequence like (w, h),
        a square crop (size, size) is made.
    scale : ``Tuple[float, float]``
        Range of size of the origin size cropped.
    ratio: ``Tuple[float, float]``
        Range of aspect ratio of the origin aspect ratio cropped.
    interpolation:
        Default: PIL.Image.BILINEAR
    min_area_frac: ``float``
        Minimal area fraction requirement of the original bounding box.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), min_area_frac=0.25, interpolation=Image.BILINEAR):
        super().__init__()
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.min_area_frac = min_area_frac

    @staticmethod
    def get_params(img, scale, ratio):
        """
        Get parameters for ``crop`` for a random sized crop.

        Parameters
        ----------
        img : ``Image``
            Image to be cropped.
        scale : ``tuple``
            Range of size of the origin size cropped.
        ratio : ``tuple``
            Range of aspect ratio of the origin aspect ratio cropped.

        Returns
        -------
        tuple
            Tarams (i, j, h, w) to be passed to ``crop`` for a random sized crop.
        """
        width, height = img.size
        area = width * height

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = width / height
        if in_ratio < min(ratio):
            w = width
            h = w / min(ratio)
        elif in_ratio > max(ratio):
            h = height
            w = h * max(ratio)
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, anns):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        new_anns = HF.resized_crop(anns, j, i, w, h, self.size, self.min_area_frac)
        if len(new_anns) == 0:
            return img, anns
        img = VF.resized_crop(img, i, j, h, w, self.size[::-1], self.interpolation)
        return img, new_anns

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4)
                                                    for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4)
                                                    for r in self.ratio))
        format_string += ', min_area_frac={0})'.format(self.min_area_frac)
        return format_string


class Resize(JointTransform):
    """Resize the image and bounding boxes.

    Parameters
    ----------
    size : ``Union[Number, Sequence[int]]``
        Desired output size. If size is a sequence like (w, h),
        the output size will be matched to this. If size is an int,
        the smaller edge of the image will be matched to this number maintaing
        the aspect ratio. i.e, if width > height, then image will be rescaled to
        (output_size * width / height, output_size)
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, img, anns):
        if img.size == self.size:
            return img, anns

        anns = HF.resize(anns, img.size, self.size)
        if isinstance(self.size, Tuple):
            size = self.size[::-1]
        else:
            size = self.size
        img = VF.resize(img, size)
        return img, anns

    def __repr__(self):
        return self.__class__.__name__ + "(size=%s)" % (self.size,)


class CenterCrop(JointTransform):
    """
    Crops the given PIL Image at the center and transform the bounding boxes.

    Parameters
    ----------
    size : ``Union[Number, Sequence[int]]``
        Desired output size of the crop. If size is an int instead of sequence like (w, h),
        a square crop (size, size) is made.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, img, anns):
        if isinstance(self.size, Tuple):
            size = self.size[::-1]
        else:
            size = self.size
        img = VF.center_crop(img, size)
        anns = HF.center_crop(anns, self.size)
        return img, anns

    def __repr__(self):
        return self.__class__.__name__ + "(size=%s)".format(self.size)


class ToPercentCoords(JointTransform):

    def __init__(self):
        super().__init__()

    def __call__(self, img, anns):
        return img, HF.to_percent_coords(anns, img.size)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToAbsoluteCoords(JointTransform):

    def __init__(self):
        super().__init__()

    def __call__(self, img, anns):
        return img, HF.to_absolute_coords(anns, img.size)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RandomHorizontalFlip(JointTransform):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, anns):
        if random.random() < self.p:
            img = VF.hflip(img)
            anns = HF.hflip(anns, img.size)
            return img, anns
        return img, anns

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(JointTransform):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, anns):
        if random.random() < self.p:
            img = VF.vflip(img)
            anns = HF.vflip(anns, img.size)
            return img, anns
        return img, anns

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def SSDTransform(size, color_jitter=True, scale=(0.1, 1), expand=(1, 4), min_area_frac=0.25):
    transforms = []
    if color_jitter:
        transforms.append(
            InputTransform(
                ColorJitter(
                    brightness=0.1, contrast=0.5,
                    saturation=0.5, hue=0.05,
                )
            )
        )
    transforms += [
        RandomApply([
            RandomExpand(expand),
        ]),
        RandomChoice([
            UseOriginal(),
            RandomSampleCrop(),
            RandomResizedCrop(size, scale=scale, ratio=(1/2, 2/1), min_area_frac=min_area_frac),
        ]),
        RandomHorizontalFlip(),
        Resize(size)
    ]
    return Compose(transforms)
