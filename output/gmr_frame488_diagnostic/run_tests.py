from pathlib import Path
import sys,unittest,importlib.util
root=Path.cwd();out=root/'output/gmr_frame488_diagnostic'
sys.path.insert(0,str(root/'realtime/humanoid_robot/src'))
import gmr_state_trajectory as p
exec(compile((out/'candidate.py').read_text(),str(out/'candidate.py'),'exec'),p.__dict__)
spec=importlib.util.spec_from_file_location('candidate_tests',out/'test_gmr_state_trajectory.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
sys.exit(not result.wasSuccessful())
