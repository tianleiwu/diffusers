import ctypes
import platform


class CudaProfiler:
    def __init__(self):
        system = platform.system()
        if system == "Linux":
            self._cudart = ctypes.cdll.LoadLibrary("libcudart.so")
        elif system == "Darwin":
            self._cudart = ctypes.cdll.LoadLibrary("libcudart.dylib")
        elif system == "Windows":
            self._cudart = ctypes.windll.LoadLibrary("cudart64_110.dll")

    def start(self):
        ret = self._cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)

    def stop(self):
        ret = self._cudart.cudaProfilerStop()
        if ret != 0:
            raise Exception("cudaProfilerStop() returned %d" % ret)
