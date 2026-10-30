import numpy as np
import time

from xmigcs.common.robot_data import RobotData
import yaml
import os
import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.special import erf
from xmigcs.utils.logging_utils import get_logger
from xmigcs.utils.xlog_utils import xlog
logger = get_logger(__name__)

class DynamicModel():
    """使用匹诺曹构建的动力学模型"""

    def __init__(self, robot_data: RobotData):
        # 加载模型
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        urdf_path = os.path.join(parent_dir, "config", "evt2_29.urdf")
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        self.robot_data_ = robot_data
        self.gravity_filter_coeff = 0.1  # Filter coefficient (adjustable)
        self.filtered_gravity = self.model.gravity.linear  # Start with standard gravity
        self.motor_num_ = 29
        self.head_motor_num_ = 2
        self.floating_base_dof_ = 6
        self.joint_index_ = np.array([15, 16, 17, 18, 19, 20, 21,
                                       22, 23, 24, 25, 26, 27, 28])

        # 滤波初始化
        self.y_prev = 0.0
        self.lambda_ = 5 * 2 * np.pi  # 10Hz
        self.dt = 0.01
        self.gamma = np.exp(-self.lambda_ * self.dt)  # 论文滤波系数
        self.beta = (1.0 - self.gamma) / self.gamma / self.dt
        self.mu_fc = np.array([70.0, 70.0])
        self.sigma_fc = np.array([25.0, 25.0])
        self.mu_hc = np.array([0.0, 0.0])
        self.sigma_hc = np.array([0.02, 0.02])
        self.offset_z = 0.88
        self.sigma_h_process = np.array([0.99, 0.99])
        self.sigma_f_measure = np.array([0.9, 0.9])
        self.liftoff_confirm_time = 0.5
        self.liftoff_duration = 0.0
        self.contact_state = np.zeros(2)
        self.contact_cov = np.ones(2)
        self.contact_threshold = np.array([0.55, 0.55])


    def compute_desired_joint_pos(self, kp_s, kd_s):
        q = self.robot_data_.q_a_[self.floating_base_dof_: -self.head_motor_num_]
        v = pin.utils.zero(self.model.nv)
        a = pin.utils.zero(self.model.nv)

        # Get the current gravity estimate from sensor
        current_gravity = self.robot_data_.get_project_gravity() * 9.81
        
        # Apply low-pass filter to smooth the gravity vector
        self.filtered_gravity = (1 - self.gravity_filter_coeff) * self.filtered_gravity + \
                               self.gravity_filter_coeff * current_gravity
        
        self.model.gravity.linear = self.filtered_gravity


        tau = pin.rnea(self.model, self.data, q, v, a)

        # Set desired torques to the computed gravity compensation torques
        tau_index = tau[-self.motor_num_:][self.joint_index_]   

        # 获取当前的kp,kd
        kp = kp_s[:-self.head_motor_num_][self.joint_index_]
        kd = kd_s[:-self.head_motor_num_][self.joint_index_]

        # 获取当前的关节位置
        q_index = self.robot_data_.q_a_[self.floating_base_dof_: -self.head_motor_num_][self.joint_index_]

        # 根据期望力矩和当前关节速度，kp，kd 计算期望位置
        q_d_index = (tau_index - kd * (0 - 0)) / kp + q_index
        return q_d_index, self.joint_index_

    def _setup_frame_indices(self, target_frame:str, source_frame:str):
        """设置需要使用的坐标系索引"""
        # 获取目标坐标系和源坐标系的索引
        try:
            self.target_frame_id = self.model.getFrameId(target_frame)        
            self.source_frame_id = self.model.getFrameId(source_frame)
        except Exception as e:
            pass
        # 检查是否成功找到坐标系
        if not hasattr(self, 'target_frame_id') or not hasattr(self, 'source_frame_id'):
            for i in range(self.model.nframes):
                pass

    def compute_stand_status(self):
        """
        使用pinocchio计算目标坐标系相对于源坐标系的变换
        将四元数转换为xyz顺序的欧拉角，并记录z轴高度
        
        Returns:
            bool: 计算成功返回True，否则返回False
        """
        try:
            # 获取当前关节角度
            # 假设 robot_data_ 中包含所有关节的当前位置
            # 需要根据实际数据结构调整
            q = np.zeros(self.model.nq)
            
            # 将实际关节位置填入q向量
            # 注意：需要根据你的机器人模型确定关节顺序
            # 这里假设前6个是浮动基座的自由度和旋转
            # 后面的关节从robot_data_中获取
            if hasattr(self.robot_data_, 'q_a_'):
                motor_q = self.robot_data_.q_a_[self.floating_base_dof_: -self.head_motor_num_]
                
                # 根据你的URDF模型设置浮动基座的初始状态
                # 如果使用浮动基座，前7个元素是基座的位姿（位置+四元数）
                if self.model.nq > self.motor_num_:
                    # 假设基座在原点，无旋转
                    q[0:3] = [0, 0, 0]  # 位置
                    q[3:7] = [1, 0, 0, 0]  # 单位四元数 (w, x, y, z)
                    # 填入电机关节位置
                    q[-self.motor_num_:] = motor_q
                else:
                    # 如果不是浮动基座，直接使用所有关节
                    q = motor_q[:self.model.nq]
            
            # 执行正向运动学
            pin.framesForwardKinematics(self.model, self.data, q)

            # 设置坐标系索引
            self._setup_frame_indices(target_frame='ankle_roll_l_link', source_frame='pelvis')

            # 获取源坐标系相对于世界坐标系的位置
            source_pose = self.data.oMf[self.source_frame_id]

            # 获取目标坐标系相对于世界坐标系的位置
            target_pose = self.data.oMf[self.target_frame_id]

            # 计算目标坐标系相对于源坐标系的变换
            # T_source_in_target = T_target_world_inv * T_source_world
            source_in_target = target_pose.actInv(source_pose)

            # 提取平移部分
            translation = source_in_target.translation

            # 提取旋转部分（旋转矩阵转四元数）
            rotation_matrix = source_in_target.rotation
            quat = pin.Quaternion(rotation_matrix)
            # pin.Quaternion的顺序是 (x, y, z, w)
            quat_xyzw = np.array([quat.x, quat.y, quat.z, quat.w])
            
            # 将四元数转换为xyz顺序的欧拉角
            # 使用scipy的Rotation
            r = R.from_quat(quat_xyzw)  # scipy使用 (x, y, z, w) 顺序
            euler = r.as_euler('xyz', degrees=False)  # 弧度制

            # 获取waist_yaw_link相对于 pelvis 的Yaw角
            self._setup_frame_indices(target_frame='pelvis', source_frame='waist_yaw_link')
            # 获取源坐标系相对于世界坐标系的位置
            source_pose_2 = self.data.oMf[self.source_frame_id]

            # 获取目标坐标系相对于世界坐标系的位置
            target_pose_2 = self.data.oMf[self.target_frame_id]

            # 计算目标坐标系相对于源坐标系的变换
            # T_source_in_target = T_target_world_inv * T_source_world
            source_in_target_2 = target_pose_2.actInv(source_pose_2)

            # 提取旋转部分（旋转矩阵转四元数）
            rotation_matrix_2 = source_in_target_2.rotation
            quat_2 = pin.Quaternion(rotation_matrix_2)
            # pin.Quaternion的顺序是 (x, y, z, w)
            quat_xyzw_2 = np.array([quat_2.x, quat_2.y, quat_2.z, quat_2.w])
            
            # 将四元数转换为xyz顺序的欧拉角
            # 使用scipy的Rotation
            r_2 = R.from_quat(quat_xyzw_2)  # scipy使用 (x, y, z, w) 顺序
            euler_2 = r_2.as_euler('xyz', degrees=False)  # 弧度制

            # 存储到robot_data_
            z_offset = 0.056
            if hasattr(self.robot_data_, 'q_a_') and len(self.robot_data_.q_a_) >= 6:
                self.robot_data_.q_a_[0] = translation[0]
                self.robot_data_.q_a_[1] = translation[1]
                self.robot_data_.q_a_[2] = translation[2] + z_offset
                self.robot_data_.q_a_[3] = euler[0]  # x轴旋转
                self.robot_data_.q_a_[4] = euler[1]  # y轴旋转
                self.robot_data_.q_a_[5] = euler_2[2]  # z轴旋转
            
            return True
            
        except Exception as e:
            return False

    def compute_desired_transform(self, target_frame:str, source_frame:str):
        """
        使用pinocchio计算目标坐标系相对于源坐标系的变换
        将四元数转换为xyz顺序的欧拉角，并记录z轴高度
        
        Returns:
            bool: 计算成功返回True，否则返回False
        """
        try:
            # 获取当前关节角度
            # 假设 robot_data_ 中包含所有关节的当前位置
            # 需要根据实际数据结构调整
            q = np.zeros(self.model.nq)
            
            # 将实际关节位置填入q向量
            # 注意：需要根据你的机器人模型确定关节顺序
            # 这里假设前6个是浮动基座的自由度和旋转
            # 后面的关节从robot_data_中获取
            if hasattr(self.robot_data_, 'q_a_'):
                motor_q = self.robot_data_.q_a_[self.floating_base_dof_: -self.head_motor_num_]
                
                # 根据你的URDF模型设置浮动基座的初始状态
                # 如果使用浮动基座，前7个元素是基座的位姿（位置+四元数）
                if self.model.nq > self.motor_num_:
                    # 假设基座在原点，无旋转
                    q[0:3] = [0, 0, 0]  # 位置
                    q[3:7] = [1, 0, 0, 0]  # 单位四元数 (w, x, y, z)
                    # 填入电机关节位置
                    q[-self.motor_num_:] = motor_q
                else:
                    # 如果不是浮动基座，直接使用所有关节
                    q = motor_q[:self.model.nq]
            
            # 执行正向运动学
            pin.framesForwardKinematics(self.model, self.data, q)

            # 设置坐标系索引
            self._setup_frame_indices(target_frame, source_frame)

            # 获取源坐标系相对于世界坐标系的位置
            source_pose = self.data.oMf[self.source_frame_id]

            # 获取目标坐标系相对于世界坐标系的位置
            target_pose = self.data.oMf[self.target_frame_id]

            # 计算目标坐标系相对于源坐标系的变换
            # T_source_in_target = T_target_world_inv * T_source_world
            source_in_target = target_pose.actInv(source_pose)

            # 提取平移部分
            translation = source_in_target.translation

            # 提取旋转部分（旋转矩阵转四元数）
            rotation_matrix = source_in_target.rotation
            quat = pin.Quaternion(rotation_matrix)
            # pin.Quaternion的顺序是 (x, y, z, w)
            quat_xyzw = np.array([quat.x, quat.y, quat.z, quat.w])
            
            # 将四元数转换为xyz顺序的欧拉角
            # 使用scipy的Rotation
            r = R.from_quat(quat_xyzw)  # scipy使用 (x, y, z, w) 顺序
            euler = r.as_euler('xyz', degrees=False)  # 弧度制
            
            return True
            
        except Exception as e:
            return False
    
    def compute_force_z(self):
        # 需要并联转串联之后再计算

        # =========================
        # 每个控制周期 k 执行以下
        # =========================

        # 1. 当前时刻传感器/计算值
        q_k = self.robot_data_.q_a_[self.floating_base_dof_: -self.head_motor_num_]
        qdot_k = self.robot_data_.q_dot_a_[self.floating_base_dof_: -self.head_motor_num_]
        M = pin.crba(self.model, self.data, q_k)
        M = np.triu(M) + np.triu(M, 1).T   # crba后补成对称矩阵
        p_k    = M @ qdot_k    # 机器动量 M * q_dot_a_
        tau_k  = self.robot_data_.tau_a_[self.floating_base_dof_: -self.head_motor_num_]    # 关节力矩 
        g_k = pin.computeGeneralizedGravity(self.model, self.data, q_k)  # 重力项
        C = pin.computeCoriolisMatrix(self.model, self.data, q_k, qdot_k) # 科里奥利矩阵
        S = np.eye(self.model.nv)

        # 2. 计算滤波输入 x_k
        x_k = self.beta * p_k + S.T @ tau_k + C.T @ qdot_k - g_k

        # 3. 你推导的正确滤波（关键！）
        y_k = self.gamma * self.y_prev + (1.0 - self.gamma) * x_k

        # 4. 公式(10)最终结果
        tau_hat_d_k = self.beta * p_k - y_k

        # 5. 更新上一时刻滤波值
        self.y_prev = y_k

        # 
        left_leg_index = [0, 1, 2, 3, 4, 5]
        right_leg_index = [6, 7, 8, 9, 10, 11]
        left_leg_torque = tau_hat_d_k[left_leg_index]
        right_leg_torque = tau_hat_d_k[right_leg_index]

        J_i_left = pin.computeFrameJacobian(
            self.model, self.data, q_k, self.model.getFrameId("ankle_roll_l_link") ,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_i_right = pin.computeFrameJacobian(
            self.model, self.data, q_k, self.model.getFrameId("ankle_roll_r_link"),
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        jv_left_leg = J_i_left[0:3, left_leg_index]
        jv_right_leg = J_i_right[0:3, right_leg_index]
        f_hat_i_left = np.linalg.pinv(jv_left_leg.T) @ left_leg_torque
        f_hat_i_right = np.linalg.pinv(jv_right_leg.T) @ right_leg_torque
        f_hat_i = np.concatenate([f_hat_i_left, f_hat_i_right])

        force_z = np.clip(np.array([f_hat_i_left[2], f_hat_i_right[2]]), 0, 400)

        return force_z

    def is_contact(self) -> bool:
        force_z = self.compute_force_z()
        force_prob = self.contact_prob_fz(force_z, self.mu_fc, self.sigma_fc)
        foot_height = self.get_foot_height()
        height_prob = self.contact_prob_hz(foot_height, self.mu_hc, self.sigma_hc)
        self.contact_state, self.contact_cov = self.kalman_height_predict(
            self.contact_state,
            self.contact_cov,
            height_prob,
            self.sigma_h_process ** 2)
        self.contact_state, self.contact_cov = self.kalman_force_update(
            self.contact_state,
            self.contact_cov,
            force_prob,
            self.sigma_f_measure ** 2)
        logger.info(
            f"contact_posterior_left: x={self.contact_state[0]}, sigma={self.contact_cov[0]}")
        logger.info(
            f"contact_posterior_right: x={self.contact_state[1]}, sigma={self.contact_cov[1]}")

        if np.all(self.contact_state < self.contact_threshold):
            self.liftoff_duration += self.dt
            if self.liftoff_duration >= self.liftoff_confirm_time:
                xlog.info(f"force_prob: {force_prob}")
                xlog.info(f"height_prob: {height_prob}")
                xlog.info(f"contact_state: {self.contact_state}")
                xlog.warning("离地")
                return False
        else:
            self.liftoff_duration = 0.0
            xlog.info("接触")

        return True
    
    def contact_prob_fz(self, f_z, mu_fc, sigma_fc):
        sigma_fc = np.maximum(sigma_fc, 1e-6)
        z = (f_z - mu_fc) / (sigma_fc * np.sqrt(2.0))
        return 0.5 * (1.0 + erf(z))
    
    def contact_prob_hz(self, z, mu_z, sigma_z):
        sigma_z = np.maximum(sigma_z, 1e-6)
        z_score = (mu_z - z) / (sigma_z * np.sqrt(2.0))
        return 0.5 * (1.0 + erf(z_score))

    def kalman_height_predict(self, x_prev, sigma_prev, u_k, sigma_w_k):
        a_k = np.zeros(2)
        b_k = np.ones(2)
        x_pred = a_k * x_prev + b_k * u_k
        sigma_pred = a_k * sigma_prev * a_k + sigma_w_k
        return x_pred, sigma_pred

    def kalman_force_update(self, x_pred, sigma_pred, z_tilde_k, sigma_v_k):
        h_k = np.ones(2)
        innovation_cov = h_k * sigma_pred * h_k + sigma_v_k
        k_k = sigma_pred * h_k / innovation_cov
        x_post = x_pred + k_k * (z_tilde_k - h_k * x_pred)
        sigma_post = (1.0 - k_k * h_k) * sigma_pred
        return x_post, sigma_post

    
    def get_foot_height(self):
        q = np.zeros(self.model.nq)
        motor_q = self.robot_data_.q_a_[self.floating_base_dof_: -self.head_motor_num_]
        if self.model.nq > self.motor_num_:
            q[0:3] = [0, 0, 0]
            q[3:7] = [1, 0, 0, 0]
            q[-self.motor_num_:] = motor_q
        else:
            q = motor_q[:self.model.nq]

        pin.framesForwardKinematics(self.model, self.data, q)

        left_foot_id = self.model.getFrameId("ankle_roll_l_link") 
        right_foot_id = self.model.getFrameId("ankle_roll_r_link")
        left_foot_pose = self.data.oMf[left_foot_id]
        right_foot_pose = self.data.oMf[right_foot_id]

        robot_quat_wxyz = self.robot_data_.get_robot_quat()
        robot_quat_xyzw = np.array([
            robot_quat_wxyz[1],
            robot_quat_wxyz[2],
            robot_quat_wxyz[3],
            robot_quat_wxyz[0],
        ])
        world_rotation = R.from_quat(robot_quat_xyzw)

        left_foot_world = world_rotation.apply(left_foot_pose.translation)
        right_foot_world = world_rotation.apply(right_foot_pose.translation)
        left_foot_z = left_foot_world[2]
        right_foot_z = right_foot_world[2]
        return np.array([left_foot_z + self.offset_z, right_foot_z + self.offset_z])