from typing import Iterable

def unique_ever_seen(iterable: Iterable) -> Iterable:
    already_seen = set()

    for item in iterable:
        if item in already_seen:
            continue
        else:
            already_seen.add(item)
            yield item


