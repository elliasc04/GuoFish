"""Every v6 test runs with the GPU hidden, subprocesses included.

"-1", not "": an empty value is dropped by PowerShell's env assignment, which
left GPUs visible during this harness's first session. CUDA reads the variable
at its (lazy) initialisation, so setting it here, before any test runs, covers
this process and every child it spawns.
"""
import os

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@pytest.fixture(scope="session", autouse=True)
def _never_initialise_cuda():
    yield
    import torch
    assert not torch.cuda.is_initialized(), "a v6 test initialised CUDA"
