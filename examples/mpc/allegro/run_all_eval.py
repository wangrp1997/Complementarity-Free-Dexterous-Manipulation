import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OBJECTS = [
    'airplane', 'binoculars', 'bowl', 'bunny', 'camera', 'can', 'cube', 'cup',
    'elephant', 'foambrick', 'mug', 'piggy_bank', 'rubber_duck', 'stick',
    'teapot', 'torus', 'water_bottle',
]
OUT = Path(__file__).resolve().parent / 'eval_results.jsonl'


def load_done():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get('returncode') == 0 and rec.get('summary'):
                done.add((rec['object'], rec['mode']))
    return done


def run_one(obj, unknown_dyn, trials=20):
    cmd = [
        sys.executable,
        str(ROOT / 'examples/mpc/allegro' / obj / 'test.py'),
        '--headless',
        f'--trials={trials}',
    ]
    if unknown_dyn:
        cmd.append('--unknown-dyn')
    print('>>', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    lines = (proc.stdout + '\n' + proc.stderr).splitlines()
    summary = next((ln for ln in lines if 'success' in ln and '/' in ln and 'trial' not in ln), '')
    rec = {
        'object': obj,
        'mode': 'unknown-dyn' if unknown_dyn else 'original',
        'returncode': proc.returncode,
        'summary': summary.strip(),
        'tail': '\n'.join(lines[-8:]),
    }
    with OUT.open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(rec['summary'] or rec['tail'], flush=True)
    return rec


if __name__ == '__main__':
    done = load_done()
    print(f'skip {len(done)} finished jobs', flush=True)
    for obj in OBJECTS:
        for unknown in (False, True):
            mode = 'unknown-dyn' if unknown else 'original'
            if (obj, mode) in done:
                print(f'skip {obj} {mode}', flush=True)
                continue
            run_one(obj, unknown)
