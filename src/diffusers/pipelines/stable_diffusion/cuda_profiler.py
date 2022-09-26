import ctypes

class CudaProfiler():
    def __init__(self):
        self._cudart = ctypes.CDLL('libcudart.so')

    def start(self):
        ret = self._cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)

    def stop(self):
        ret = self._cudart.cudaProfilerStop()
        if ret != 0:
            raise Exception("cudaProfilerStop() returned %d" % ret)
