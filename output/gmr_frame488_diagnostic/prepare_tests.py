from pathlib import Path
root=Path.cwd();out=root/'output/gmr_frame488_diagnostic'
test=(root/'tests/test_gmr_state_trajectory.py').read_text()
test=test.replace('    def test_solver_keeps_nonzero_initial_velocity_and_acceleration(self):', '''    def test_feasible_turnaround_is_not_rejected_by_global_control_bounds(self):
        # q(t)=t-t^2 stays below .25, but its global degree-5 Bernstein
        # control points reach .3. The old fixed-row check rejects high=.26.
        start = JointState(np.array([0.]), np.array([1.]), np.array([-2.]))
        refs = [JointState(np.array([t-t*t]), np.array([1-2*t]), np.array([-2.]))
                for t in np.linspace(.1, 1., 10)]
        bridge = solve_horizon(start, refs, 1., np.array([-2.]), np.array([.26]), (),
                               np.r_[np.zeros(3),1.,np.zeros(3),0.], projection_settings())
        self.assertIsNotNone(bridge)
        self.assertTrue(bridge.within_limits(0, np.array([-2.]), np.array([.26])))
        for field in ('position', 'velocity', 'acceleration'):
            np.testing.assert_allclose(getattr(bridge.at_time(0), field), getattr(start, field), atol=1e-12)

    def test_subinterval_controls_reconstruct_the_same_polynomial(self):
        from gmr_state_trajectory import subdivided_bernstein_matrix
        from math import comb
        for degree in range(3, 6):
            power = np.arange(degree+1, dtype=float) * (-1.)**np.arange(degree+1)
            controls = (subdivided_bernstein_matrix(degree) @ power).reshape(4, degree+1)
            for segment in range(4):
                for u in np.linspace(0., 1., 11):
                    value = sum(controls[segment,k]*comb(degree,k)*u**k*(1-u)**(degree-k)
                                for k in range(degree+1))
                    expected = np.polynomial.polynomial.polyval((segment+u)/4, power)
                    self.assertAlmostEqual(value, expected, places=12)

    def test_solver_keeps_nonzero_initial_velocity_and_acceleration(self):''')
(out/'test_gmr_state_trajectory.py').write_text(test)
(out/'run_tests.py').write_text('''from pathlib import Path
import sys,unittest,importlib.util
root=Path.cwd();out=root/'output/gmr_frame488_diagnostic'
sys.path.insert(0,str(root/'realtime/humanoid_robot/src'))
import gmr_state_trajectory as p
exec(compile((out/'candidate.py').read_text(),str(out/'candidate.py'),'exec'),p.__dict__)
spec=importlib.util.spec_from_file_location('candidate_tests',out/'test_gmr_state_trajectory.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
sys.exit(not result.wasSuccessful())
''')
r=(out/'test_candidate.py').read_text().replace("d=np.load(out/'guide.npz')", "d=np.load(root/'output/gmr_frame69_diagnostic/guide.npz')")
(out/'regression_previous_clip.py').write_text(r)
