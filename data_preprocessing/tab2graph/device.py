"""Device related utilities."""
from collections import namedtuple
import os
import torch

DeviceInfo = namedtuple('DeviceInfo', ['cpu_count', 'gpu_devices'])

def get_device_info():
    if torch.cuda.is_available():
        gpu_devices = [f"cuda:{devid}" for devid in range(torch.cuda.device_count())]
    else:
        gpu_devices = []
    cpu_count = int(os.environ.get("NUM_VISIBLE_CPUS", os.cpu_count()))
    return DeviceInfo(cpu_count, gpu_devices)
