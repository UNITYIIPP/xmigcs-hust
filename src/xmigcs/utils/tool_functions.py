import numpy as np
from scipy.spatial.transform import Rotation

def quat_rotate_inverse_numpy(q, v):
    """
    q: [x, y, z, w], shape=(4)\\
    v: [x, y, z], shape=(3)
    """
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def euler_to_quaternion_scipy(roll, pitch, yaw, degrees=False):
    """
    使用SciPy进行欧拉角转四元数
    参数:
        roll: 绕x轴的旋转角度
        pitch: 绕y轴的旋转角度
        yaw: 绕z轴的旋转角度
        degrees: 输入角度是否为度，默认为弧度
    返回:
        [w, x, y, z]: 四元数分量 (w为实部)
    """
    # 创建旋转对象 (顺序: 'xyz' 对应 roll, pitch, yaw)
    rotation = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=degrees)

    # 转换为四元数 (顺序: [x, y, z, w])
    quaternion = rotation.as_quat()

    return [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # 返回 w, x, y, z

def print_colored(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    colored_text = "\033[93m" + sep.join(str(arg) for arg in args) + "\033[0m"  # 黄色
    # colored_text = "\033[91m" + sep.join(str(arg) for arg in args) + "\033[0m"  # 红色

def wrap_to_pi(angles: np.ndarray):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

def heading_to_yaw(current_heading, target_heading, kp=0.5, yaw_limit=1.0):
    return np.clip(kp * wrap_to_pi(target_heading - current_heading), -yaw_limit, yaw_limit)
