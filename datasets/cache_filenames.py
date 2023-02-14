import json
from functools import wraps
from torchvision.datasets import ImageFolder
import argparse
from pathlib import Path


def file_cache(filename):
    """Decorator to cache the output of a function to disk."""
    def decorator(f):
        @wraps(f)
        def decorated(self, directory, *args, **kwargs):
            filepath = Path(directory) / filename
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                out = f(self, directory, *args, **kwargs)
                filepath.write_text(json.dumps(out))
            return out
        return decorated
    return decorator

class CachedImageFolder(ImageFolder):
    @file_cache(filename="cached_classes.json")
    def find_classes(self, directory, *args, **kwargs):
        classes = super().find_classes(directory, *args, **kwargs)
        return classes

    @file_cache(filename="cached_structure.json")
    def make_dataset(self, directory, *args, **kwargs):
        dataset = super().make_dataset(directory, *args, **kwargs)
        return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    target_dir = Path(args.path)
    if not target_dir.exists():
        print("The target directory doesn't exist")
        raise SystemExit(1)

    print("Initialising Image Folder caching...")
    dataset = CachedImageFolder(target_dir)
    print("Image Folder caching successful!")




if __name__ == '__main__':
    main()
    