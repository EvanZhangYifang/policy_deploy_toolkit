"""Check exported mixed actor against checkpoint-derived test vectors.

Run from any directory with Python, NumPy and ONNX Runtime installed:
    python verify_mixed_policy.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

bundle = Path(__file__).resolve().parent
metadata = json.loads((bundle / 'metadata.json').read_text())
vectors = json.loads((bundle / 'test_vectors.json').read_text())['tests']
path = bundle / 'policy.onnx'
actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
if actual_sha != metadata['policy_onnx_sha256']:
    raise RuntimeError(f'ONNX SHA-256 mismatch: {actual_sha}')
session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
for case in vectors:
    obs = np.asarray([case['obs']], dtype=np.float32)
    expected = np.asarray([case['raw_action']], dtype=np.float32)
    actual = session.run([metadata['onnx']['output_name']], {metadata['onnx']['input_name']: obs})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"{case['mode'].upper()}: PASS (maximum absolute error {np.max(np.abs(actual-expected)):.3g})")
print('ONNX hash and all checkpoint-derived inference vectors passed.')
