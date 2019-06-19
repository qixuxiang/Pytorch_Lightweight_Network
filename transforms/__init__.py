import time
import random
import torchvision.transforms.functional as VF


class Transform:

    def __init__(self):
        pass

    def __call__(self, input, target):
        pass

    def __repr__(self):
        return pprint(self)


class JointTransform(Transform):

    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

    def __call__(self, input, target):
        return self.transform(input, target)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class InputTransform(Transform):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, input, target):
        return self.transform(input), target

    # def __repr__(self):
    #     return pprint(self)
        # format_string = self.__class__.__name__ + '('
        # format_string += '\n'
        # format_string += '    {0}'.format(self.transform)
        # format_string += '\n)'
        # return format_string


class TargetTransform(Transform):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, input, target):
        return input, self.transform(target)

    # def __repr__(self):
    #     return pprint(self)


class Compose(Transform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            # start = time.time()
            img, target = t(img, target)
            # print("%.4f" % ((time.time() - start) * 1000))
        return img, target

    # def __repr__(self):
    #     return pprint(self)
        # format_string = self.__class__.__name__ + '('
        # for t in self.transforms:
        #     format_string += '\n'
        #     format_string += '    {0}'.format(t)
        # format_string += '\n)'
        # return format_string


class UseOriginal(Transform):
    """Use the original image and annotations.
    """

    def __init__(self):
        pass

    def __call__(self, img, target):
        return img, target


class RandomApply(Transform):

    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            for t in self.transforms:
                img, target = t(img, target)
        return img, target

    # def __repr__(self):
    #     return pprint(self)
        # format_string = self.__class__.__name__ + '('
        # for t in self.transforms:
        #     format_string += '\n'
        #     format_string += '    {0}'.format(t)
        # format_string += '\n)'
        # return format_string


class RandomChoice(Transform):
    """Apply single transformation randomly picked from a list.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.RandomChoice([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        t = random.choice(self.transforms)
        img, target = t(img, target)
        return img, target

    # def __repr__(self):
    #     return pprint(self)
        # format_string = self.__class__.__name__ + '('
        # for t in self.transforms:
        #     format_string += '\n'
        #     format_string += '    {0}'.format(t)
        # format_string += '\n)'
        # return format_string


class ToTensor(JointTransform):

    def __init__(self):
        super().__init__()

    def __call__(self, img, anns):
        return VF.to_tensor(img), anns


def pprint(t, level=0, sep='    '):
    pre = sep * level
    if not isinstance(t, Transform) or isinstance(t, JointTransform):
        return pre + repr(t)
    format_string = pre + type(t).__name__ + '('
    if hasattr(t, 'transforms'):
        for t in getattr(t, 'transforms'):
            format_string += '\n'
            format_string += pprint(t, level + 1)
        format_string += '\n'
        format_string += pre + ')'
    elif hasattr(t, 'transform'):
        format_string += '\n'
        format_string += pprint(getattr(t, 'transform'), level + 1)
        format_string += '\n'
        format_string += pre + ')'
    else:
        format_string += ')'
    return format_string

