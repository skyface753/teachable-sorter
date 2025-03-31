from rotpy.system import SpinSystem
from rotpy.camera import CameraList
system = SpinSystem()

cameras = CameraList.create_from_system(system, True, True)
print(cameras.get_size())
