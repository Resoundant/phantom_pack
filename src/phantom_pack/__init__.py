"""Tools for finding and analyzing Phantom Pack vials in DICOM images."""

from importlib.metadata import PackageNotFoundError, version

from .phantom_pack import find_packs_in_images, load_and_label, phantom_pack, process_directory

try:
    __version__ = version("phantom-pack")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "find_packs_in_images",
    "load_and_label",
    "phantom_pack",
    "process_directory",
]
