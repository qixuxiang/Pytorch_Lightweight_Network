__ACTIVATION__ = 'relu'
__NORM_LAYER__ = 'bn'


def get_default_activation():
    global __ACTIVATION__
    return __ACTIVATION__


def set_default_activation(name):
    global __ACTIVATION__
    __ACTIVATION__ = name


def get_default_norm_layer():
    global __NORM_LAYER__
    return __NORM_LAYER__


def set_default_norm_layer(name):
    global __NORM_LAYER__
    __NORM_LAYER__ = name
