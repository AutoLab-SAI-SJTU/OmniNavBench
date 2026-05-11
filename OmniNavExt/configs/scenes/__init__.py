import inspect
from typing import Type, List

from omni.metropolis.utils.config_file.section import Section
from . import defines


def get_all_section_cls() -> List[Type[Section]]:
    section_cls = []
    for member_tuple in inspect.getmembers(defines, predicate=inspect.isclass):
        name, cls_obj = member_tuple
        if cls_obj.__module__ == defines.__name__ and issubclass(cls_obj, Section):
            section_cls.append(cls_obj)
    return section_cls
