"""
改进的混合傅里叶-多项式激励轨迹优化

改进点:
1. 直接计算观测矩阵 W 的条件数 κ(W)（而非 κ(W^T·W)）
2. 适应度函数: 指数衰减 exp(-log10(κ)/2)，单调递减（条件数越小越好）
3. 适应度权重: 条件数40% + 基础激励10% + 位置约束30% + 速度约束10% + 加速度多样性10%
4. 周期性由结构保证，不计入适应度（原5%权重转移至条件数）
5. 工作空间覆盖已达目标，不计入适应度（原5%权重转移至条件数）
6. 文献参考 (Swevers 1997): κ(W) ≈ 100 是实际可达到的良好值
"""

import numpy as np
import json
import datetime
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False


def clean_for_json(obj):
    """递归清理对象中的NaN和Inf值，使其可以被JSON序列化"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return 1e15 if obj > 0 else -1e15
        else:
            return float(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    else:
        return obj


def safe_json_dump(data, filename, verbose=True):
    """
    安全地保存JSON文件，包含完整的错误处理和数据清理
    
    Args:
        data: 要保存的数据
        filename: 文件名
        verbose: 是否打印详细信息
    
    Returns:
        (bool, str): (是否成功, 消息)
    """
    try:
        if verbose:
            print(f"\n[保存] 开始清理数据...")
        cleaned_data = clean_for_json(data)
        
        if verbose:
            print(f"[保存] 验证数据可序列化...")
        json_str = json.dumps(cleaned_data, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"[保存] 数据大小: {len(json_str):,} 字符")
            print(f"[保存] 写入文件: {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_str)
            f.flush()
        
        if verbose:
            print(f"[保存] 验证文件完整性...")
        with open(filename, 'r', encoding='utf-8') as f:
            verify_data = json.load(f)
        
        if verbose:
            print(f"[保存] ✓ 成功!")
        return True, f"成功保存到 {filename}"
        
    except json.JSONDecodeError as e:
        error_msg = f"JSON编码错误: {e}"
        print(f"[保存] ✗ {error_msg}")
        
        # 尝试保存为纯文本
        backup_filename = filename.replace('.json', '_raw.txt')
        try:
            with open(backup_filename, 'w', encoding='utf-8') as f:
                f.write(str(data))
            return False, f"JSON失败，已保存纯文本到 {backup_filename}"
        except:
            return False, error_msg
        
    except Exception as e:
        error_msg = f"保存失败: {str(e)}"
        print(f"[保存] ✗ {error_msg}")
        print(f"[保存] 错误类型: {type(e).__name__}")
        
        # 打印详细错误信息
        import traceback
        traceback.print_exc()
        
        # 尝试保存到备份文件
        backup_filename = filename.replace('.json', '_backup.json')
        try:
            print(f"\n[保存] 尝试保存到备份文件...")
            with open(backup_filename, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
                f.flush()
            return True, f"已保存到备份文件: {backup_filename}"
        except Exception as e2:
            return False, f"主文件和备份都失败: {str(e)}, {str(e2)}"


class CombinedParameters:
    def __init__(self):
        self.l2 = 0.4
        self.l3 = 0.4
        self.l4 = 0.04
        self.l5 = 0.135
        self.g = 9.81


class HybridTrajectoryImproved:
    """
    改进的混合轨迹（方案C: offset可优化）
    
    优化变量:
    - q_offset (3个): 每个关节的位置偏移量（可优化）
    - a_l, b_l (30个): 傅里叶系数
    总计: 33个变量
    """
    def __init__(self, coefficients, combined_params, N=5, t_f=10.0, n_points=1000, offset_mode='optimize'):
        """
        Args:
            coefficients: [q_offset_1, q_offset_2, q_offset_3,
                          a_1^1,...,a_N^1, b_1^1,...,b_N^1,
                          a_1^2,...,a_N^2, b_1^2,...,b_N^2,
                          a_1^3,...,a_N^3, b_1^3,...,b_N^3]
                          共 3 + 3×2N = 3 + 30 = 33 个系数
            offset_mode: 'optimize' - offset可优化（方案C）
        """
        self.coefficients = np.array(coefficients)
        self.params = combined_params
        self.N = N
        self.t_f = t_f
        self.n_points = n_points
        self.omega_f = 2 * np.pi / t_f
        self.offset_mode = offset_mode
        
        # 提取位置偏移量
        self.q_offset = self.coefficients[:3]
        
        # 提取傅里叶系数
        fourier_coeffs = self.coefficients[3:]
        self.joint_coeffs = fourier_coeffs.reshape(3, 2*N)
        
        self.a = np.zeros((3, N))
        self.b = np.zeros((3, N))
        self.c = np.zeros((3, 6))
        
        for joint in range(3):
            self.a[joint, :] = self.joint_coeffs[joint, 0:N]
            self.b[joint, :] = self.joint_coeffs[joint, N:2*N]
            self._calculate_polynomial_coefficients(joint)
        
        self.fitness = None
    
    def _calculate_polynomial_coefficients(self, joint):
        a = self.a[joint, :]
        b = self.b[joint, :]
        
        c_0 = 0.0
        c_1 = 0.0
        c_2 = 0.0
        
        for l in range(1, self.N + 1):
            c_0 += b[l-1] / (self.omega_f * l)
            c_1 -= a[l-1]
            c_2 -= 0.5 * b[l-1] * self.omega_f * l
        
        t_f = self.t_f
        c_3 = (-2 * c_2 * t_f - 10 * c_1) / (t_f ** 2)
        c_4 = (15 * c_1 + c_2 * t_f) / (t_f ** 3)
        c_5 = -6 * c_1 / (t_f ** 4)
        
        self.c[joint, :] = [c_0, c_1, c_2, c_3, c_4, c_5]
    
    def generate_trajectory(self, num_cycles=1):
        total_time = self.t_f * num_cycles
        t = np.linspace(0, total_time, self.n_points * num_cycles)
        n = len(t)
        
        q = np.zeros((3, n))
        q_dot = np.zeros((3, n))
        q_ddot = np.zeros((3, n))
        
        for idx, t_i in enumerate(t):
            if t_i == 0:
                j = 1
            else:
                j = int(np.ceil(t_i / self.t_f))
            
            t_rel = t_i - (j - 1) * self.t_f
            
            for joint in range(3):
                fourier_q = 0
                fourier_q_dot = 0
                fourier_q_ddot = 0
                
                for l in range(1, self.N + 1):
                    a_l = self.a[joint, l-1]
                    b_l = self.b[joint, l-1]
                    omega_l = self.omega_f * l
                    
                    fourier_q += (a_l / omega_l) * np.sin(omega_l * t_i)
                    fourier_q -= (b_l / omega_l) * np.cos(omega_l * t_i)
                    
                    fourier_q_dot += a_l * np.cos(omega_l * t_i)
                    fourier_q_dot += b_l * np.sin(omega_l * t_i)
                    
                    fourier_q_ddot += -a_l * omega_l * np.sin(omega_l * t_i)
                    fourier_q_ddot += b_l * omega_l * np.cos(omega_l * t_i)
                
                poly_q = 0
                poly_q_dot = 0
                poly_q_ddot = 0
                
                for k in range(6):
                    c_k = self.c[joint, k]
                    poly_q += c_k * (t_rel ** k)
                    if k >= 1:
                        poly_q_dot += c_k * k * (t_rel ** (k-1))
                    if k >= 2:
                        poly_q_ddot += c_k * k * (k-1) * (t_rel ** (k-2))
                
                # 添加位置偏移量
                q[joint, idx] = fourier_q + poly_q + self.q_offset[joint]
                q_dot[joint, idx] = fourier_q_dot + poly_q_dot
                q_ddot[joint, idx] = fourier_q_ddot + poly_q_ddot
        
        return q, q_dot, q_ddot, t
    
    def build_observation_matrix_combined(self, q, q_dot, q_ddot):
        """
        构建16参数的W_minimal观测矩阵
        （与verify_16params_observation_matrix.py中的实现相同）
        """
        n_points = q.shape[1]
        W_full = np.zeros((3 * n_points, 16))
        
        q1, q2, q3 = q[0, :], q[1, :], q[2, :]
        dq1, dq2, dq3 = q_dot[0, :], q_dot[1, :], q_dot[2, :]
        ddq1, ddq2, ddq3 = q_ddot[0, :], q_ddot[1, :], q_ddot[2, :]
        
        l2, l3, l4, l5, g = self.params.l2, self.params.l3, self.params.l4, self.params.l5, self.params.g
        
        for i in range(n_points):
            q1_i, q2_i, q3_i = q1[i], q2[i], q3[i]
            dq1_i, dq2_i, dq3_i = dq1[i], dq2[i], dq3[i]
            ddq1_i, ddq2_i, ddq3_i = ddq1[i], ddq2[i], ddq3[i]
            
            # 三角函数
            c2, s2 = np.cos(q2_i), np.sin(q2_i)
            c3, s3 = np.cos(q3_i), np.sin(q3_i)
            c23 = np.cos(q2_i + q3_i)
            s23 = np.sin(q2_i + q3_i)
            c2_2 = c2**2
            c3_2 = c3**2
            sin_2q2 = np.sin(2*q2_i)
            sin_2q3 = np.sin(2*q3_i)
            
            # === W_minimal第1行 (关节1力矩方程) ===
            row1 = 3 * i
            W_full[row1, 0] = ddq1_i
            W_full[row1, 1] = dq1_i*dq2_i*sin_2q2 - ddq1_i*(c2_2 - 1)
            W_full[row1, 2] = ddq1_i*c2_2 - dq1_i*dq2_i*sin_2q2
            W_full[row1, 3] = dq1_i*dq3_i*sin_2q3 - ddq1_i*(c3_2 - 1)
            W_full[row1, 4] = ddq1_i*c3_2 - dq1_i*dq3_i*sin_2q3
            W_full[row1, 5] = ddq1_i*l3**2*c3_2 - dq1_i*dq3_i*l3**2*sin_2q3
            W_full[row1, 6] = ddq1_i*l2**2*c2_2 - dq1_i*dq2_i*l2**2*sin_2q2
            
            # 列7: 复杂表达式
            W_full[row1, 7] = (ddq1_i*(l2**2*c2_2 + l3**2*c3_2 + l4**2 + l5**2 + 2*l2*l4*c2 + 2*l3*l4*c3 + 2*l2*l3*c2*c3)
                              - dq1_i*dq2_i*l2**2*sin_2q2 - dq1_i*dq3_i*l3**2*sin_2q3 
                              + ddq2_i*l2*l5*s2 + ddq3_i*l3*l5*s3 + dq2_i**2*l2*l5*c2 + dq3_i**2*l3*l5*c3 
                              + 2*l2*l3*c2*c3 - 2*dq1_i*dq2_i*l2*l4*s2 - 2*dq1_i*dq3_i*l3*l4*s3 
                              - 2*dq1_i*dq2_i*l2*l3*c3*s2 - 2*dq1_i*dq3_i*l2*l3*c2*s3)
            
            W_full[row1, 8] = 2*dq1_i*dq2_i*l3*c3*s2 - 2*ddq1_i*l3*c2*c3 + 2*dq1_i*dq3_i*l3*c2*s3
            W_full[row1, 9] = 2*ddq1_i*l2*c2*c3 - 2*dq1_i*dq2_i*l2*c3*s2 - 2*dq1_i*dq3_i*l2*c2*s3
            W_full[row1, 10] = ddq1_i*c2_2 - dq1_i*dq2_i*sin_2q2
            W_full[row1, 11] = ddq1_i*c3_2 - dq1_i*dq3_i*sin_2q3
            W_full[row1, 12:16] = 0
            
            # === W_minimal第2行 (关节2力矩方程) ===
            row2 = 3 * i + 1
            W_full[row2, 0] = 0
            W_full[row2, 1] = -dq1_i**2*c2*s2
            W_full[row2, 2] = dq1_i**2*c2*s2
            W_full[row2, 3:5] = 0
            W_full[row2, 5] = 0
            W_full[row2, 6] = c2*s2*dq1_i**2*l2**2 + ddq2_i*l2**2 + g*c2*l2
            
            # 列7: 复杂表达式
            W_full[row2, 7] = (ddq2_i*l2**2 + g*l2*c2 + dq1_i**2*s2*(l2*l4 + l2**2*c2 + l2*l3*c3)
                              - ddq3_i*l2*l3*c23 + ddq1_i*l2*l5*s2 + dq3_i**2*l2*l3*s23)
            
            W_full[row2, 8] = -l3*c3*s2*dq1_i**2 - l3*s23*dq3_i**2 + g*c2 + ddq3_i*l3*c23
            W_full[row2, 9] = l2*c3*s2*dq1_i**2 + l2*s23*dq3_i**2 - ddq3_i*l2*c23
            W_full[row2, 10] = c2*s2*dq1_i**2 + ddq2_i
            W_full[row2, 11] = 0
            W_full[row2, 12] = ddq2_i
            W_full[row2, 13] = g*c2
            W_full[row2, 14:16] = 0
            
            # === W_minimal第3行 (关节3力矩方程) ===
            row3 = 3 * i + 2
            W_full[row3, 0:3] = 0
            W_full[row3, 3] = -dq1_i**2*c3*s3
            W_full[row3, 4] = dq1_i**2*c3*s3
            W_full[row3, 5] = c3*s3*dq1_i**2*l3**2 + ddq3_i*l3**2 + g*c3*l3
            W_full[row3, 6] = 0
            
            # 列7: 复杂表达式
            W_full[row3, 7] = (ddq3_i*l3**2 - g*l3*c3 + dq1_i**2*s3*(l3*l4 + l3**2*c3 + l2*l3*c2)
                              - ddq2_i*l2*l3*c23 + ddq1_i*l3*l5*s3 + dq2_i**2*l2*l3*s23)
            
            W_full[row3, 8] = -l3*c2*s3*dq1_i**2 - l3*s23*dq2_i**2 + ddq2_i*l3*c23
            W_full[row3, 9] = l2*c2*s3*dq1_i**2 + l2*s23*dq2_i**2 - g*c3 - ddq2_i*l2*c23
            W_full[row3, 10] = 0
            W_full[row3, 11] = c3*s3*dq1_i**2 + ddq3_i
            W_full[row3, 12:14] = 0
            W_full[row3, 14] = ddq3_i
            W_full[row3, 15] = g*c3
        
        return W_full
    
    def _calculate_workspace_coverage_reward(self, q):
        """工作空间覆盖奖励"""
        # 缩小5°安全余量后的范围
        allowed_ranges = [np.deg2rad(170.0), np.deg2rad(80.0), np.deg2rad(80.0)]
        
        coverage_score = 0.0
        for joint in range(3):
            q_max = np.max(q[joint, :])
            q_min = np.min(q[joint, :])
            q_range = q_max - q_min
            
            coverage_ratio = q_range / allowed_ranges[joint]
            optimal_coverage = 0.60
            reward = np.exp(-5 * (coverage_ratio - optimal_coverage)**2)
            coverage_score += reward
        
        coverage_score /= 3
        return coverage_score
    
    def _calculate_acceleration_diversity_reward(self, q_ddot):
        """加速度多样性奖励"""
        diversity_score = 0.0
        
        for joint in range(3):
            acc = q_ddot[joint, :]
            
            # 时域多样性
            acc_std = np.std(acc)
            acc_mean = np.mean(np.abs(acc)) + 1e-6
            temporal_diversity = acc_std / acc_mean
            temporal_diversity = np.tanh(temporal_diversity)
            
            # 频域多样性
            fft_acc = np.fft.fft(acc)
            fft_mag = np.abs(fft_acc[:self.N+1])
            power = fft_mag**2
            power_norm = power / (np.sum(power) + 1e-10)
            entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
            freq_diversity = entropy / np.log(self.N + 1)
            
            diversity_score += 0.5 * temporal_diversity + 0.5 * freq_diversity
        
        diversity_score /= 3
        return diversity_score
    
    def calculate_fitness(self):
        """
        方案C适应度函数（offset可优化）
        """
        q, q_dot, q_ddot, t = self.generate_trajectory(num_cycles=1)
        
        # 位置惩罚 - 新约束空间（缩小5°安全余量）: [95°, 265°], [95°, 175°], [185°, 265°]
        joint_limits = [(np.deg2rad(95.0), np.deg2rad(265.0)), 
                       (np.deg2rad(95.0), np.deg2rad(175.0)), 
                       (np.deg2rad(185.0), np.deg2rad(265.0))]
        range_limits = [np.deg2rad(170.0), np.deg2rad(80.0), np.deg2rad(80.0)]
        
        position_penalty = 0.0
        for joint in range(3):
            q_min, q_max = joint_limits[joint]
            violations_low = np.maximum(0, q_min - q[joint, :])
            violations_high = np.maximum(0, q[joint, :] - q_max)
            position_penalty += np.sum(violations_low**2) + np.sum(violations_high**2)
        
        position_penalty_normalized = position_penalty / (self.n_points * 3)
        
        # 速度惩罚 - ±80 deg/s
        v_limit_rad = np.deg2rad(80.0)
        velocity_penalty = 0.0
        for joint in range(3):
            v_violations_low = np.maximum(0, -v_limit_rad - q_dot[joint, :])
            v_violations_high = np.maximum(0, q_dot[joint, :] - v_limit_rad)
            velocity_penalty += np.sum(v_violations_low**2) + np.sum(v_violations_high**2)
        
        velocity_penalty_normalized = velocity_penalty / (self.n_points * 3)
        
        # 构建观测矩阵并计算条件数
        W = self.build_observation_matrix_combined(q, q_dot, q_ddot)
        
        try:
            cond_num = np.linalg.cond(W)  # 直接计算 κ(W)，而非 κ(W^T·W)
            if np.isnan(cond_num) or np.isinf(cond_num):
                cond_num = 1e10
        except:
            cond_num = 1e10
        
        # 1. 条件数（40%）- 单调递减函数（条件数越小越好）
        # 文献参考 (Swevers 1997): κ(W) ≈ 100 是实际可达到的较好值
        # 优化目标：κ(W) 越小越好，理想情况 κ(W) → 1
        log_cond = np.log10(max(cond_num, 1))
        fitness_cond = np.exp(-log_cond / 2.0)  # 指数衰减，单调递减
        fitness_cond = np.clip(fitness_cond, 0, 1) * 0.40  # 40%权重（周期性5%+工作空间5%转移而来）
        
        # 2. 基础激励（10%）
        vel_rms = np.sqrt(np.mean(q_dot**2))
        acc_rms = np.sqrt(np.mean(q_ddot**2))
        excitation = vel_rms + 0.1 * acc_rms
        fitness_excitation = np.tanh(excitation / 5.0) * 0.10
        
        # 3. 位置约束（30%）
        fitness_position = -np.minimum(position_penalty_normalized * 500, 1.0) * 0.30
        
        # 4. 速度约束（10%）
        fitness_velocity = -0.10 * (1 - np.exp(-50 * velocity_penalty_normalized))
        
        # 5. 周期性（仅用于诊断，不计入适应度）
        # 周期性由混合傅里叶-多项式结构天然保证，不需要优化权重
        pos_error = np.sum((q[:, 0] - q[:, -1])**2)
        vel_error = np.sum((q_dot[:, 0] - q_dot[:, -1])**2)
        acc_error = np.sum((q_ddot[:, 0] - q_ddot[:, -1])**2)
        periodicity_error = np.sqrt(pos_error + vel_error + 0.1 * acc_error)
        # fitness_periodicity = 0.0  # 不计入适应度（权重已转移至条件数）
        
        # 6. 工作空间覆盖（仅用于诊断，不计入适应度）
        # 实际覆盖率通常能达到60%以上，不需要优化权重
        workspace_coverage_score = self._calculate_workspace_coverage_reward(q)  # 仅计算，不加权
        # workspace_coverage_reward = 0.0  # 权重已转移至条件数
        
        # 7. 加速度多样性奖励（10%）
        acceleration_diversity_reward = self._calculate_acceleration_diversity_reward(q_ddot) * 0.10
        
        # 总适应度 = 100%
        # 条件数40% + 基础激励10% + 位置惩罚30% + 速度惩罚10% + 加速度多样性10%
        # 注：周期性由结构保证，不计入适应度（原5%权重转移至条件数）
        # 注：工作空间覆盖已达目标，不计入适应度（原5%权重转移至条件数）
        self.fitness = (fitness_cond + fitness_excitation + fitness_position + 
                       fitness_velocity +
                       acceleration_diversity_reward)
        
        # 诊断信息
        try:
            WTW = W.T @ W
            determinant = np.linalg.det(WTW)
            eigenvalues = np.linalg.eigvalsh(WTW)
            min_eigenval = np.min(eigenvalues[eigenvalues > 0]) if np.any(eigenvalues > 0) else 0
        except:
            determinant = 0
            min_eigenval = 0
        
        # 工作空间覆盖率
        workspace_coverage_ratios = []
        for joint in range(3):
            q_range = np.max(q[joint, :]) - np.min(q[joint, :])
            coverage_ratio = q_range / range_limits[joint]
            workspace_coverage_ratios.append(float(coverage_ratio))
        
        # 速度利用率
        velocity_utilization = vel_rms / v_limit_rad
        
        # 加速度多样性指标
        acc_diversity_per_joint = []
        for joint in range(3):
            acc_std = np.std(q_ddot[joint, :])
            acc_mean = np.mean(np.abs(q_ddot[joint, :])) + 1e-6
            diversity = acc_std / acc_mean
            acc_diversity_per_joint.append(float(diversity))
        
        self.diagnostics = {
            'feasible': bool(position_penalty < 0.01 and velocity_penalty < 0.01),
            'condition_number': float(cond_num),
            'position_penalty': float(position_penalty),
            'velocity_penalty': float(velocity_penalty),
            'periodicity_error': float(periodicity_error),
            'velocity_rms_deg': float(np.rad2deg(vel_rms)),
            'acceleration_rms_deg': float(np.rad2deg(acc_rms)),
            'workspace_coverage_ratios': workspace_coverage_ratios,
            'workspace_coverage_mean': float(np.mean(workspace_coverage_ratios)),
            'velocity_utilization': float(velocity_utilization),
            'acceleration_diversity_per_joint': acc_diversity_per_joint,
            'acceleration_diversity_mean': float(np.mean(acc_diversity_per_joint)),
            'determinant': float(determinant),
            'min_eigenvalue': float(min_eigenval),
            'q_offset_rad': [float(x) for x in self.q_offset],
            'q_offset_deg': [float(np.rad2deg(x)) for x in self.q_offset],
            'initial_position': [float(q[i, 0]) for i in range(3)],
            'initial_velocity': [float(q_dot[i, 0]) for i in range(3)],
            'final_position': [float(q[i, -1]) for i in range(3)],
            'final_velocity': [float(q_dot[i, -1]) for i in range(3)],
            'fitness_components': {
                'cond': float(fitness_cond),
                'excite': float(fitness_excitation),
                'pos': float(fitness_position),
                'vel': float(fitness_velocity),
                'acceleration_diversity': float(acceleration_diversity_reward)
                # 注：周期性由结构保证，不计入适应度
                # 注：工作空间覆盖已达目标，不计入适应度
            },
            'constraints': {
                'position_limits': [[95, 265], [95, 175], [185, 265]],  # 缩小5°安全余量
                'velocity_limit_deg': 80.0
            }
        }
        
        return self.fitness


class HarrisHawksOptimization:
    def __init__(self, objective_function, dim, bounds, num_hawks=50, max_iter=200, combined_params=None, save_dir='incremental_results'):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.num_hawks = num_hawks
        self.max_iter = max_iter
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(num_hawks, dim))
        self.fitness = np.zeros(num_hawks)
        self.best_position = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.combined_params = combined_params
        self.start_time = None
        self.save_dir = save_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"\n[系统] 创建增量保存目录: {save_dir}")
        
    def optimize(self):
        self.start_time = time.time()
        
        # 初始化种群
        print("\n初始化种群...")
        for i in range(self.num_hawks):
            self.fitness[i] = self.objective_function(self.positions[i])
        
        best_idx = np.argmax(self.fitness)
        self.best_position = self.positions[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.fitness_history.append(self.best_fitness)
        
        # 初始状态
        avg_fitness = np.mean(self.fitness)
        self._print_progress(0, avg_fitness)
        
        for t in range(1, self.max_iter + 1):
            E1 = 2 * (1 - t / self.max_iter)
            
            for i in range(self.num_hawks):
                E0 = 2 * np.random.random() - 1
                E = E1 * E0
                
                if abs(E) >= 1:
                    q = np.random.random()
                    rand_idx = np.random.randint(0, self.num_hawks)
                    
                    if q >= 0.5:
                        r1 = np.random.random(self.dim)
                        r2 = np.random.random(self.dim)
                        new_pos = self.positions[rand_idx] - r1 * np.abs(self.positions[rand_idx] - 2 * r2 * self.positions[i])
                    else:
                        r3 = np.random.random(self.dim)
                        r4 = np.random.random(self.dim)
                        avg_pos = np.mean(self.positions, axis=0)
                        new_pos = (self.best_position - avg_pos) - r3 * (self.bounds[:, 0] + r4 * (self.bounds[:, 1] - self.bounds[:, 0]))
                else:
                    r = np.random.random()
                    
                    if r >= 0.5 and abs(E) >= 0.5:
                        Delta_X = self.best_position - self.positions[i]
                        new_pos = Delta_X - E * np.abs(np.random.random(self.dim) * self.best_position - self.positions[i])
                    elif r >= 0.5 and abs(E) < 0.5:
                        new_pos = self.best_position - E * np.abs(self.best_position - self.positions[i])
                    elif r < 0.5 and abs(E) >= 0.5:
                        Y = self.best_position - E * np.abs(np.random.random(self.dim) * self.best_position - self.positions[i])
                        Y = np.clip(Y, self.bounds[:, 0], self.bounds[:, 1])
                        fit_Y = self.objective_function(Y)
                        if fit_Y > self.fitness[i]:
                            new_pos = Y
                        else:
                            S = np.random.random(self.dim) * self.positions[i]
                            new_pos = Y + np.random.random(self.dim) * S
                    else:
                        Y = self.best_position - E * np.abs(np.random.random(self.dim) * self.best_position - np.mean(self.positions, axis=0))
                        Y = np.clip(Y, self.bounds[:, 0], self.bounds[:, 1])
                        fit_Y = self.objective_function(Y)
                        if fit_Y > self.fitness[i]:
                            new_pos = Y
                        else:
                            S = np.random.random(self.dim) * self.positions[i]
                            new_pos = Y + np.random.random(self.dim) * S
                
                new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
                new_fitness = self.objective_function(new_pos)
                
                if new_fitness > self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fitness
                    
                    if new_fitness > self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = new_pos.copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # 详细进度显示
            if t % 10 == 0 or t == self.max_iter:
                avg_fitness = np.mean(self.fitness)
                self._print_progress(t, avg_fitness)
        
        # 最终总结
        elapsed_time = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"优化完成！总用时: {elapsed_time/60:.1f} 分钟")
        print(f"{'='*80}\n")
        
        return self.best_position, self.best_fitness
    
    def _print_progress(self, generation, avg_fitness):
        """打印详细的优化进度并保存增量结果"""
        elapsed = time.time() - self.start_time
        
        # 基本信息
        progress = generation / self.max_iter * 100
        print(f"\n{'─'*80}")
        print(f"[代数 {generation:3d}/{self.max_iter}] 进度: {progress:5.1f}% | 用时: {elapsed/60:5.1f}min", end="")
        
        # 预计剩余时间
        if generation > 0:
            avg_time_per_gen = elapsed / generation
            remaining = avg_time_per_gen * (self.max_iter - generation)
            print(f" | 预计剩余: {remaining/60:5.1f}min")
        else:
            print()
        
        # 适应度信息
        print(f"  最优适应度: {self.best_fitness:+.6f} | 平均适应度: {avg_fitness:+.6f} | 差值: {self.best_fitness-avg_fitness:.6f}")
        
        # 如果有 combined_params，计算并显示详细诊断
        diag = None
        if self.combined_params is not None:
            try:
                best_traj = HybridTrajectoryImproved(self.best_position, self.combined_params, N=5, offset_mode='optimize')
                best_traj.calculate_fitness()
                diag = best_traj.diagnostics
                
                print(f"  条件数: {diag['condition_number']:,.1f} | ", end="")
                print(f"速度RMS: {diag['velocity_rms_deg']:.2f} deg/s | ", end="")
                print(f"工作空间: {diag['workspace_coverage_mean']*100:.1f}%")
                
                # 位置和速度约束检查
                pos_ok = "[OK]" if diag['position_penalty'] < 0.01 else "[!]"
                vel_ok = "[OK]" if diag['velocity_penalty'] < 0.01 else "[!]"
                print(f"  位置惩罚: {diag['position_penalty']:.4f} {pos_ok} | ", end="")
                print(f"速度惩罚: {diag['velocity_penalty']:.4f} {vel_ok} | ", end="")
                print(f"可行性: {'✓' if diag['feasible'] else '✗'}")
                
                # Offset值
                print(f"  Offset: [{diag['q_offset_deg'][0]:.1f}°, {diag['q_offset_deg'][1]:.1f}°, {diag['q_offset_deg'][2]:.1f}°]")
                    
                # 收敛趋势
                if len(self.fitness_history) >= 20:
                    recent_improvement = self.fitness_history[-1] - self.fitness_history[-20]
                    if abs(recent_improvement) < 1e-6:
                        print(f"  收敛状态: 已收敛 (近20代改进 < 1e-6)")
                    else:
                        print(f"  收敛状态: 持续改进 (近20代改进: {recent_improvement:+.6f})")
            except Exception as e:
                print(f"  诊断失败: {str(e)}")
                pass  # 如果诊断失败，跳过
        
        print(f"{'─'*80}")
        
        # 保存增量结果
        self._save_incremental_result(generation, diag)
    
    def _save_incremental_result(self, generation, diagnostics=None):
        """保存增量结果到JSON文件"""
        try:
            results = {
                'generation': generation,
                'timestamp': datetime.datetime.now().isoformat(),
                'best_coefficients': [float(x) for x in self.best_position],
                'best_fitness': float(self.best_fitness),
                'fitness_history': [float(f) for f in self.fitness_history],
                'elapsed_time_minutes': (time.time() - self.start_time) / 60,
                'progress_percent': (generation / self.max_iter) * 100
            }
            
            if diagnostics is not None:
                results['diagnostics'] = diagnostics
            
            filename = os.path.join(self.save_dir, f'gen_{generation:04d}_{self.timestamp}.json')
            success, message = safe_json_dump(results, filename, verbose=False)
            
            if success:
                print(f"  [保存] ✓ 增量: gen_{generation:04d}")
            else:
                print(f"  [保存] ✗ 失败")
        except Exception as e:
            print(f"  [保存] ✗ 异常: {str(e)}")


def create_bounds(N=5, fourier_scale=None):
    """
    创建边界（方案C: offset可优化）
    
    变量: [q_offset (3), a_l (15), b_l (15)]
    
    Args:
        N: 傅里叶谐波数
        fourier_scale: 傅里叶系数缩放因子列表 [joint1, joint2, joint3]
                      默认: [0.60, 0.15, 0.10] (方案2: 关节2增至0.15)
    """
    if fourier_scale is None:
        fourier_scale = [0.6, 0.30, 0.30]
    
    bounds = []
    
    # q_offset 边界（方案1: 扩大搜索范围 + 新约束空间）
    # 新约束: [95°,265°], [95°,175°], [185°,265°]
    bounds.extend([
        (np.deg2rad(160.0), np.deg2rad(240.0)),   # 关节1: 160°-240°（扩大上限+20°）
        (np.deg2rad(100.0), np.deg2rad(170.0)),   # 关节2: 100°-170°（保持不变）
        (np.deg2rad(210.0), np.deg2rad(270.0))    # 关节3: 210°-270°（扩大上限+10°）
    ])
    
    # 处理 fourier_scale：可以是单值或列表
    if isinstance(fourier_scale, (list, tuple)):
        scales = fourier_scale
    else:
        scales = [fourier_scale] * 3
    
    # 傅里叶系数 - 每个关节使用独立的scale
    for joint in range(3):
        joint_scale = scales[joint]
        # a 系数
        for l in range(1, N+1):
            scale = joint_scale / l
            bounds.append((-scale, scale))
        # b 系数
        for l in range(1, N+1):
            scale = joint_scale / l
            bounds.append((-scale, scale))
    
    return bounds


def run_optimization():
    # 方案1+2+安全余量参数
    joint_scales = [0.60, 0.15, 0.10]  # 方案2: 关节2从0.10增至0.15
    
    print("=" * 80)
    print("混合傅里叶-多项式激励轨迹优化 - 方案1+2+安全余量")
    print("=" * 80)
    
    print("\n🚀 优化策略:")
    print("  ✓ 方案1: 扩大offset搜索范围 (关节1,3上限扩大)")
    print("  ✓ 方案2: 增大关节2的fourier_scale (0.10 → 0.15)")
    print("  ✓ 安全余量: 约束范围各缩小5°")
    
    print("\n约束配置:")
    print("  1. offset可优化:")
    print("     - 关节1: [160°, 240°]")
    print("     - 关节2: [100°, 170°]")
    print("     - 关节3: [210°, 270°]")
    print("  2. 位置约束（缩小5°安全余量）:")
    print("     - 关节1: [95°, 265°]")
    print("     - 关节2: [95°, 175°]")
    print("     - 关节3: [185°, 265°]")
    print("  3. 速度限位: ±80 deg/s")
    print("  4. 分关节fourier_scale: [0.60, 0.15, 0.10]")
    print("  5. 适应度权重: 条件数40% + 基础激励10% + 位置30% + 速度10%")
    print("  6. 奖励项: 加速度多样性10%")
    print("  7. 优化目标: 最小化 κ(W)（越小越好，文献参考值≈100）")
    print("  注: 周期性由结构保证，不计入适应度")
    print("  注: 工作空间覆盖已达目标，不计入适应度")
    
    print("\n优化变量: 33个")
    print("  - q_offset: 3个 (可优化)")
    print("  - 傅里叶系数: 30个 (3关节 × 5谐波 × 2)")
    
    print(f"\n每个关节的fourier_scale:")
    print(f"  关节1: {joint_scales[0]}")
    print(f"  关节2: {joint_scales[1]} (方案2: 增大以提升覆盖率)")
    print(f"  关节3: {joint_scales[2]}")
    
    combined_params = CombinedParameters()
    
    def objective_func(coeffs):
        trajectory = HybridTrajectoryImproved(coeffs, combined_params, N=5, offset_mode='optimize')
        return trajectory.calculate_fitness()
    
    bounds = create_bounds(N=5, fourier_scale=joint_scales)
    
    print(f"\n开始优化...")
    print(f"  种群大小: 50")
    print(f"  最大迭代: 200")
    print(f"  增量保存: 启用 (每10代保存一次)")
    print(f"  保存目录: incremental_results/\n")
    
    hho = HarrisHawksOptimization(
        objective_function=objective_func,
        dim=33,
        bounds=bounds,
        num_hawks=50,
        max_iter=200,
        combined_params=combined_params
    )
    
    best_coeffs, best_fitness = hho.optimize()
    best_trajectory = HybridTrajectoryImproved(best_coeffs, combined_params, N=5, offset_mode='optimize')
    best_trajectory.calculate_fitness()
    
    print("=" * 80)
    print("最终结果")
    print("=" * 80)
    
    diag = best_trajectory.diagnostics
    print(f"\n总适应度: {best_fitness:.6f}")
    print(f"可行性: {'✓ 是' if diag['feasible'] else '✗ 否'}")
    
    print(f"\n核心指标:")
    cond = diag['condition_number']
    if cond < 50:
        cond_status = '✓ 优秀'
    elif cond < 200:
        cond_status = '✓ 良好'
    elif cond < 1000:
        cond_status = '○ 可接受'
    else:
        cond_status = '⚠ 较差'
    print(f"  条件数 κ(W): {cond:.2e} {cond_status}")
    print(f"               (目标: 越小越好，文献参考值 ≈100)")
    print(f"  位置惩罚: {diag['position_penalty']:.6f} {'[OK]' if diag['position_penalty'] < 0.01 else '[!]'}")
    print(f"  速度惩罚: {diag['velocity_penalty']:.6f} {'[OK]' if diag['velocity_penalty'] < 0.01 else '[!]'}")
    print(f"  周期性误差: {diag['periodicity_error']:.6f}")
    
    print(f"\n激励指标:")
    print(f"  速度RMS: {diag['velocity_rms_deg']:.2f} deg/s (限位: 80 deg/s)")
    print(f"  加速度RMS: {diag['acceleration_rms_deg']:.2f} deg/s²")
    print(f"  速度利用率: {diag['velocity_utilization']*100:.1f}%")
    
    print(f"\n工作空间覆盖:")
    cov_ratios = diag['workspace_coverage_ratios']
    print(f"  关节1: {cov_ratios[0]*100:.1f}% ([95°,265°])")
    print(f"  关节2: {cov_ratios[1]*100:.1f}% ([95°,175°])")
    print(f"  关节3: {cov_ratios[2]*100:.1f}% ([185°,265°])")
    print(f"  平均覆盖率: {diag['workspace_coverage_mean']*100:.1f}% (目标: 60%)")
    
    print(f"\n加速度多样性:")
    acc_div = diag['acceleration_diversity_per_joint']
    print(f"  关节1: {acc_div[0]:.3f}")
    print(f"  关节2: {acc_div[1]:.3f}")
    print(f"  关节3: {acc_div[2]:.3f}")
    print(f"  平均多样性: {diag['acceleration_diversity_mean']:.3f}")
    
    print(f"\n最优Offset值:")
    for i in range(3):
        print(f"  关节{i+1}: {diag['q_offset_deg'][i]:7.3f}°")
    
    print(f"\n适应度分解:")
    comp = diag['fitness_components']
    cond = diag['condition_number']
    print(f"  条件数:       {comp['cond']:+.6f} (40%) ← 主目标：最小化 κ(W)")
    print(f"                κ(W)={cond:.1f}, 适应度=exp(-log10({cond:.1f})/2)*0.40")
    print(f"  基础激励:     {comp['excite']:+.6f} (10%)")
    print(f"  位置约束:     {comp['pos']:+.6f} (30%) ← 强制满足约束")
    print(f"  速度约束:     {comp['vel']:+.6f} (10%) ← 避免速度超限")
    print(f"  --- 辅助奖励项 ---")
    print(f"  加速度多样性: {comp['acceleration_diversity']:+.6f} (10%) ← 帮助降低条件数")
    print(f"  总和:         {sum(comp.values()):+.6f}")
    print(f"  ")
    print(f"  注: 周期性由结构保证 (误差={diag['periodicity_error']:.2e})，不计入适应度")
    print(f"  注: 工作空间覆盖={diag['workspace_coverage_mean']*100:.1f}%，已达目标，不计入适应度")
    
    results = {
        'best_coefficients': [float(x) for x in best_coeffs],
        'best_fitness': float(best_fitness),
        'diagnostics': diag,
        'fitness_history': [float(f) for f in hho.fitness_history],
        'parameters': {
            'method': 'hybrid_fourier_polynomial_improved_scheme_c',
            'fourier_harmonics': 5,
            'offset_mode': 'optimize',
            'num_variables': 33,
            'num_hawks': 50,
            'max_iter': 200,
            'n_points': 1000,
            'fourier_scale': joint_scales,
            'offset_bounds': [[160, 240], [100, 170], [210, 270]],
            'position_limits_deg': [[95, 265], [95, 175], [185, 265]],
            'velocity_limit_deg': 80.0,
            'fitness_weights': {
                'cond': 0.40,
                'excite': 0.10,
                'pos': 0.30,
                'vel': 0.10,
                'acceleration_diversity': 0.10
                # 周期性由结构保证，不计入适应度（权重0%）
                # 工作空间覆盖已达目标，不计入适应度（权重0%）
            }
        }
    }
    
    filename = f'hybrid_improved_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    success, message = safe_json_dump(results, filename)
    
    print("\n" + "=" * 80)
    if success:
        print(f"✓ 结果已成功保存: {filename}")
    else:
        print(f"✗ 保存失败: {message}")
    print("=" * 80)


if __name__ == "__main__":
    run_optimization()



