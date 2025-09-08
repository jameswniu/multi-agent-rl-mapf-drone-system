# Simple load test example: check heavy loop runs under a time limit

import time

def test_load_under_pressure():
    start = time.time()
    for _ in range(100000):
        _ = 2 * 2
    elapsed = time.time() - start
    assert elapsed < 5
