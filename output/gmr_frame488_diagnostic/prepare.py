from pathlib import Path
root=Path.cwd();out=root/'output/gmr_frame488_diagnostic';src=root/'realtime/humanoid_robot/src/gmr_state_trajectory.py'
s=src.read_text()
s=s.replace('def _affine(start, duration):', '''def subdivided_bernstein_matrix(degree, subdivisions=4):
    """Bound each subinterval independently without loosening physical limits."""
    blocks = []
    for segment in range(subdivisions):
        origin = segment / subdivisions
        width = 1. / subdivisions
        restriction = np.array([[math.comb(k, j) * origin**(k-j) * width**j
                                 if k >= j else 0. for k in range(degree+1)]
                                for j in range(degree+1)])
        blocks.append(bernstein_matrix(degree) @ restriction)
    return np.vstack(blocks)


def _affine(start, duration):''')
s=s.replace('transform = bernstein_matrix(5-derivative)', 'transform = subdivided_bernstein_matrix(5-derivative)')
(out/'candidate.py').write_text(s)
r=(out/'check.py').read_text().replace("source=(base/'src/gmr_state_trajectory.py').read_text()", "source=(out/'candidate.py').read_text()")
(out/'test_candidate.py').write_text(r)
