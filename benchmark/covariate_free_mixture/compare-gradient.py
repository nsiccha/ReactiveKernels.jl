import ast, math, re, sys
from pathlib import Path

def load(path):
    s = Path(path).read_text()
    m = re.fullmatch(r'\(value = ([^,]+), gradient = (\[.*?\]), point = (\[.*?\])\)\s*', s)
    if m is None:
        raise ValueError(f'Unrecognized gradient dump: {path}')
    return float(m[1]), ast.literal_eval(m[2]), ast.literal_eval(m[3])

base, actual = map(load, sys.argv[1:])
assert base[2] == actual[2], 'points differ'
assert len(base[1]) == len(actual[1]), 'gradient lengths differ'
assert math.isclose(base[0], actual[0], rel_tol=2e-13, abs_tol=2e-13), 'values differ'
assert all(math.isclose(x, y, rel_tol=2e-13, abs_tol=2e-13) for x,y in zip(base[1], actual[1])), 'gradients differ'
print('GRADIENT_COMPARE point_identical=true value_abs_error=', abs(base[0]-actual[0]),
      'max_gradient_abs_error=', max((abs(x-y) for x,y in zip(base[1],actual[1])), default=0.0))
