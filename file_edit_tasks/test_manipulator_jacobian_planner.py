# test_manipulator_jacobian_planner.py
import unittest
import math
from manipulator_jacobian_planner import Manipulator

class TestManipulatorJacobianPlanner(unittest.TestCase):
    def test_inverse_velocity_zero_twist(self):
        m = Manipulator([1.0, 1.0])
        q = [0.1, -0.2]
        v_task = [0.0, 0.0, 0.0]
        qdot = m.inverse_velocity(q, v_task, damping=0.01, nullspace_gain=0.0, q_null=None)
        # With zero task and no nullspace, qdot should be zero vector
        self.assertTrue(all(abs(v) < 1e-12 for v in qdot))

    def test_nullspace_projection_moves_toward_null(self):
        m = Manipulator([1.0, 1.0])
        q = [0.5, -0.5]
        q_null = [0.0, 0.0]
        # Zero task but positive nullspace gain should produce qdot moving toward q_null
        qdot = m.inverse_velocity(q, [0.0, 0.0, 0.0], damping=0.01, nullspace_gain=0.5, q_null=q_null)
        # Direction toward null is (q_null - q)
        dir_to_null = [q_null[i] - q[i] for i in range(2)]
        # Dot product should be positive (moving toward)
        dot = dir_to_null[0]*qdot[0] + dir_to_null[1]*qdot[1]
        self.assertGreater(dot, 0.0)

    def test_ik_resolved_rate_reaches_goal(self):
        # 2-link planar manipulator with equal 1m links. Start at straight configuration -> end at (2,0)
        m = Manipulator([1.0, 1.0])
        q_init = [0.0, 0.0]
        # Goal: reach (1.0, 1.0) with orientation ~ 0.0
        goal = (1.0, 1.0, 0.0)
        q_final, traj = m.ik_resolved_rate(goal, q_init, dt=0.05, max_iters=2000, tol=1e-3,
                                           damping=0.05, nullspace_gain=0.0)
        x, y, th = m.forward_kinematics(q_final)
        err = math.hypot(x - goal[0], y - goal[1])
        self.assertLess(err, 1e-3)

if __name__ == '__main__':
    unittest.main()