from alfred.fusion.geometry import quaternion_to_euler
from alfred.fusion.geometry import euler_to_quaternion


yaw, pitch, roll = 0.12, 0, 0
q = euler_to_quaternion(yaw, pitch, roll)
print(q)
angles = quaternion_to_euler(q[0], q[1], q[2], q[3])
print(angles)