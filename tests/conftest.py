import pytest
import torch

blackwell = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def pytest_collection_modifyitems(config, items):
    if not blackwell:
        skip = pytest.mark.skip(reason="needs a Blackwell (sm_100+) GPU")
        for item in items:
            item.add_marker(skip)
