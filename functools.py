from toolz.curried import curry, isiterable, map, filter


def lmap(f, *iterables):
    return list(map(f, *iterables))


@curry
def recursive_lmap(f, iterable):
    if isiterable(next(iter(iterable))):
        return lmap(recursive_lmap(f), iterable)
    else:
        return lmap(f, iterable)


@curry
def find(f, seq):
    try:
        return next(filter(lambda x: f(x[1]), enumerate(seq)))[0]
    except StopIteration:
        return None
