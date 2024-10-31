from numbers import Number

def get_nd_tuple(scalar_or_tuple, d):
    if isinstance(scalar_or_tuple, Number):
        return (scalar_or_tuple,) * d
    else:
        return scalar_or_tuple

