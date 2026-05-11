import ast
import importlib
import os
from pathlib import Path
from OmniNavExt.envset.settings import WriterSetting
from omni.replicator.core import WriterRegistry
import carb


def _get_all_built_in_writers() -> dict[str, "IRABasicWriter"]:
    """
    Get all built-in writers in this extension in writers directory.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    files = os.listdir(dir_path)
    writer_class_dict = {}

    for file in files:
        if not file.endswith(".py") or file in {"utils.py", "__init__.py"}:
            continue  # Skip non-writer files

        module_name = f"OmniNavExt.data_generation.writers.{file.replace('.py', '')}"

        try:
            importlib.import_module(module_name)  # Import writer module
        except ImportError as e:
            print(f"Skipping {module_name} due to ImportError: {e}")
            continue  # Skip if module fails to load

        file_path = Path(dir_path).joinpath(file)
        with open(file_path, "r", encoding="utf-8") as f:
            module_node = ast.parse(f.read())

        for node in ast.walk(module_node):
            if isinstance(node, ast.ClassDef):
                try:

                    # instead of get writer directly
                    # fetch exist writer dict
                    writer_dict = WriterRegistry.get_writers()
                    # get node name
                    node_name = node.name
                    # check whether the node is an avaialble writer
                    writer = writer_dict.get(node_name, None)
                    if writer is None:
                        continue

                    writer_module = importlib.import_module(
                        "OmniNavExt.data_generation.writers.writer"
                    )
                    # get IRABasicWriter module
                    IRABasicWriter = getattr(writer_module, "IRABasicWriter", None)
                    # check whether the target class inherit the IRABasicWriter
                    if IRABasicWriter and issubclass(writer, IRABasicWriter):
                        # record the target class in the dictionary
                        writer_class_dict[node_name] = writer
                except Exception as e:
                    print(f"Error processing writer {node.name}: {e}")
                    continue  # Skip problematic writers

    return writer_class_dict


def get_writers_params_values():
    """
    Get all writers parameter names and default values.
    """
    writer_dict = {}
    writer_class_dict = _get_all_built_in_writers()
    for name, writer in writer_class_dict.items():
        writer_dict[name] = writer.params_values()
    return writer_dict


def get_writers_tooltips():

    writer_dict = {}

    writer_dict["BasicWriter"] = "Replicator BasicWriter."

    writer_class_dict = _get_all_built_in_writers()
    for name, writer in writer_class_dict.items():
        writer_dict[name] = writer.tooltip()
    return writer_dict


def get_writers_allow_basic_writers_params():

    writer_dict = {}

    writer_dict["BasicWriter"] = True

    writer_class_dict = _get_all_built_in_writers()
    writer_class_dict = _get_all_built_in_writers()
    for name, writer in writer_class_dict.items():
        writer_dict[name] = writer.allow_basic_writer_params()
    return writer_dict
