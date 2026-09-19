"""Apply the tested DAQP precision fix only while the default v2 batch is idle."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
base=root/'realtime/humanoid_robot'
sys.path.insert(0,str(base/'src'))
from gmr_batch_lock import batch_lock, BatchAlreadyRunning
path=base/'src/gmr_state_trajectory.py'
old=b'solution = solve_qp(H, f, np.vstack(rows), np.concatenate(bounds), solver=settings["solver"])'
new=b'''# DAQP defaults permit residuals larger than the exact curve validator.
        solver_options = {"primal_tol": 1e-9, "dual_tol": 1e-9} if settings["solver"] == "daqp" else {}
        solution = solve_qp(H, f, np.vstack(rows), np.concatenate(bounds), solver=settings["solver"], **solver_options)'''
try:
    with batch_lock(base/'data/aistpp_gmr_v2/.batch.lock'):
        data=path.read_bytes()
        newline=b'\r\n' if b'\r\n' in data else b'\n'
        replacement=new.replace(b'\n',newline)
        if replacement in data:
            print('Solver precision fix is already installed.')
        elif data.count(old)!=1:
            raise RuntimeError('Planner source changed; refusing to patch an unexpected version.')
        else:
            backup=Path(__file__).with_name('gmr_state_trajectory.before_solver_fix.py')
            if backup.exists():
                raise RuntimeError('Backup already exists; inspect it before retrying.')
            backup.write_bytes(data)
            updated=data.replace(old,replacement)
            compile(updated,str(path),'exec')
            path.write_bytes(updated)
            print('Installed tighter DAQP tolerances. Validation thresholds are unchanged.')
            print('Rerun build_gmr_v2.py with --resume. Changed implementation fingerprints trigger rebuilding.')
except BatchAlreadyRunning as exc:
    print(str(exc))
    print('No generator source was changed. Run this installer again after the batch finishes.')
    sys.exit(2)
