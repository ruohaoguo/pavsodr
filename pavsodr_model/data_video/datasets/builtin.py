import os

from .pavsodr import (
    register_pavsodr_instances,
    _get_pavsodr_instances_meta,
)

# ==== Predefined splits for PAVSODR ===========
_PREDEFINED_SPLITS_PAVSODR = {
    "pavsodr_train": ("pavsodr/train/JPEGImages",
                         "pavsodr/train.json"),
}

def register_all_pavsodr(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PAVSODR.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_pavsodr_instances(
            key,
            _get_pavsodr_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_pavsodr(_root)


