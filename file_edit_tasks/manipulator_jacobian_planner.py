"""
Planar revolute manipulator Jacobian-based resolved-rate planner.
Implements FK, Jacobian, damped least-squares pseudoinverse, nullspace projection,
and a simple iterative IK (resolved-rate control).
"""
import math
import copy

def mat_transpose(A):
    return [list(row) for row in zip(*A)]

def mat_mul(A, B):
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            if aik:
                for j in range(p):
                    C[i][j] += aik * B[k][j]
    return C

def mat_add(A, B):
    m = len(A)
    n = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]

def mat_identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def mat_inverse(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate([r[:] for r in A])]
    for i in range(n):
        pivot = i
        maxval = abs(M[i][i])
        for r in range(i + 1, n):
            if abs(M[r][i]) < maxval:
                maxval = abs(M[r][i])
                pivot = r
        if maxval < 1e-12:
            raise ValueError('Matrix is singular and cannot be inverted')
        if pivot != i:
            M[i], M[pivot] = (M[pivot], M[i])
        piv = M[i][i]
        M[i] = [val / piv for val in M[i]]
        for r in range(n):
            if r == i:
                continue
            fac = M[r][i]
            if fac:
                M[r] = [M[r][c] - fac * M[i][c] for c in range(2 * n)]
    return [row[n:] for row in M]

def pseudoinverse_damped(J, damping):
    JT = mat_transpose(J)
    JJt = mat_mul(J, JT)
    m = len(JJt)
    damp = [[JJt[i][j] + (damping ** 2 if i == j else 0.0) for j in range(m)] for i in range(m)]
    inv = mat_inverse(damp)
    return mat_mul(JT, inv)

class Manipulator:

    def __init__(self, link_lengths):
        """
        Initialize planar manipulator with given link lengths (list of positive floats).
        Joints are revolute about Z; base at origin.
        """
        self.link_lengths = list(link_lengths)
        self.n = len(self.link_lengths)

    def joint_positions(self, q):
        positions = [(0.0, 0.0)]
        x = 0.0
        y = 0.0
        theta = 0.0
        for i in range(self.n):
            theta += q[i]
            x += self.link_lengths[i] * math.cos(theta)
            y += self.link_lengths[i] * math.sin(theta)
            positions.append((x, y))
        return positions

    def forward_kinematics(self, q):
        theta = 0.0
        x = 0.0
        y = 0.0
        for i in range(self.n):
            theta += q[i]
            x += self.link_lengths[i] * math.cos(theta)
            y += self.link_lengths[i] * math.sin(theta)
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        return (x, y, theta)

    def jacobian(self, q):
        """
        3 x n Jacobian for planar revolute manipulator:
        Jv_i = z x (p_end - p_i) = (- (y_e - y_i), x_e - x_i)
        Jw_i = 1 (rotation about z)
        """
        positions = self.joint_positions(q)
        p_end = positions[-1]
        m = 3
        n = self.n
        J = [[0.0] * n for _ in range(m)]
        for i in range(n):
            px, py = positions[i]
            rx = p_end[0] - px
            ry = p_end[1] - py
            J[0][i] = -ry
            J[1][i] = rx
            J[2][i] = 1.0
        return J

    def inverse_velocity(self, q, v_task, damping=0.01, nullspace_gain=0.0, q_null=None):
        """
        Compute joint velocities to achieve v_task (3-vector) and a nullspace motion toward q_null.
        v_task: [vx, vy, omega]
        q_null: desired joint configuration for nullspace, if provided.
        """
        J = self.jacobian(q)
        J_pinv = pseudoinverse_damped(J, damping)
        v_col = [[v_task[0]], [v_task[1]], [v_task[2]]]
        qdot_primary = [row[0] for row in mat_mul(J_pinv, v_col)]
        if nullspace_gain and q_null is not None:
            Jprod = mat_mul(J_pinv, J)
            I = mat_identity(self.n)
            N = [[I[i][j] - Jprod[i][j] for j in range(self.n)] for i in range(self.n)]
            dq = [q_null[i] - q[i] for i in range(self.n)]
            dq_scaled = [[nullspace_gain * val] for val in dq]
            qdot_null = [row[0] for row in mat_mul(N, dq_scaled)]
        else:
            qdot_null = [0.0] * self.n
        qdot = [qdot_primary[i] + qdot_null[i] for i in range(self.n)]
        return qdot

    def ik_resolved_rate(self, goal_pose, q_init, dt=0.05, max_iters=1000, tol=0.0001, damping=0.05, nullspace_gain=0.0, q_null=None, Kp_pos=1.0, Kp_ori=1.0):
        """
        Iteratively integrate qdot to drive end-effector to goal_pose = (x, y, theta).
        Returns final q and trajectory list.
        Uses proportional error in task space to form desired twist v_task = K * error.
        """
        q = list(q_init)
        traj = [list(q)]
        for _ in range(max_iters):
            x, y, th = self.forward_kinematics(q)
            ex = goal_pose[0] - x
            ey = goal_pose[1] - y
            etheta = (goal_pose[2] - th + math.pi) % (2 * math.pi) - math.pi
            err_pos = math.hypot(ex, ey)
            if err_pos < tol and abs(etheta) < tol:
                break
            vx = Kp_pos * ex
            vy = Kp_pos * ey
            omega = Kp_ori * etheta
            vmax = 1.0
            vnorm = math.hypot(vx, vy)
            if vnorm > vmax:
                scale = vmax / vnorm
                vx *= scale
                vy *= scale
            v_task = [vx, vy, omega]
            qdot = self.inverse_velocity(q, v_task, damping=damping, nullspace_gain=nullspace_gain, q_null=q_null)
            q = [q[i] + qdot[i] * dt for i in range(self.n)]
            traj.append(list(q))
        return (q, traj)