"""Locate raw source datasets that live outside this (disk-lean) app tree.

The large raw datasets (cifar-100-binary/, CIFAR-100-C/, ...) are NOT kept in
apps/vggnet to save space -- only the small compiled kernel/*.bin products are.
The originals live in ~/vggnet. When a prep script is asked for a raw input
that isn't present relative to the current directory, resolve it against that
external root instead of failing.

Resolution order for a given raw path P:
  1. P as-is, if it exists (absolute paths and real local files keep working);
  2. $VGGNET_DATA_ROOT/P, if the env var is set;
  3. ~/vggnet/P  (the default location of the originals).

Override the root with:  export VGGNET_DATA_ROOT=/some/where
"""

import os
import sys

DEFAULT_ROOT = os.path.expanduser("~/vggnet")


def data_root():
    return os.path.expanduser(os.environ.get("VGGNET_DATA_ROOT", DEFAULT_ROOT))


def resolve_raw(path):
    """Return an existing path for a raw dataset input, searching the external
    data root as a fallback. Exit with a clear message if nothing is found."""
    # 1. exactly as given (absolute, or present in the cwd)
    if os.path.exists(path):
        return path

    # 2/3. under the external data root (env override, else ~/vggnet)
    candidate = os.path.join(data_root(), path)
    if os.path.exists(candidate):
        print(f"[dataset_root] '{path}' not local; using {candidate}")
        return candidate

    sys.stderr.write(
        f"error: raw dataset '{path}' not found.\n"
        f"  looked in: {os.path.abspath(path)}\n"
        f"         and: {candidate}\n"
        f"  The raw datasets are not stored in apps/vggnet (disk space); the\n"
        f"  originals live in {data_root()}. Set VGGNET_DATA_ROOT to point\n"
        f"  elsewhere, or pass an absolute path.\n")
    sys.exit(1)
