import torch
import pytest


def get_test_devices() -> dict[str, torch.device]:
    """Create a dictionary with the devices to test the source code. CUDA devices will be test only in case the
    current hardware supports it.

    Return:
        dict(str, torch.device): list with devices names.
    """
    devices: dict[str, torch.device] = {}
    devices["cpu"] = torch.device("cpu")
    if torch.cuda.is_available():
        devices["cuda"] = torch.device("cuda:0")
    return devices


def get_test_dtypes() -> dict[str, torch.dtype]:
    """Create a dictionary with the dtypes the source code.

    Return:
        dict(str, torch.dtype): list with dtype names.
    """
    dtypes: dict[str, torch.dtype] = {}
    dtypes["bfloat16"] = torch.bfloat16
    dtypes["float16"] = torch.float16
    dtypes["float32"] = torch.float32
    dtypes["float64"] = torch.float64
    return dtypes


TEST_DEVICES: dict[str, torch.device] = get_test_devices()
TEST_DTYPES: dict[str, torch.dtype] = get_test_dtypes()


@pytest.fixture()
def device(request) -> torch.device:
    return TEST_DEVICES[request.param]


@pytest.fixture()
def dtype(request) -> torch.dtype:
    return TEST_DTYPES[request.param]


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu", help="device to run tests on (cpu or cuda)")
    parser.addoption(
        "--dtype", action="store", default="float32", help="dtype to run tests on (bfloat16, float16, float32, float64)"
    )


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        devices = get_test_devices()
        device_option = metafunc.config.getoption("device")
        if device_option == "all":
            devices = list(devices.keys())
        else:
            devices = [device_option]
        metafunc.parametrize("device", devices, indirect=True)

    if "dtype" in metafunc.fixturenames:
        dtypes = get_test_dtypes()
        dtype_option = metafunc.config.getoption("dtype")
        if dtype_option == "all":
            dtypes = list(dtypes.keys())
        else:
            dtypes = [dtype_option]
        metafunc.parametrize("dtype", dtypes, indirect=True)
