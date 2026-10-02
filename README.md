# hyperloglog_counter

A small, dependency-free HyperLogLog counter for approximating the number of
distinct elements in a data stream.

## Usage

```python
from hyperloglog_counter import HyperLogLog

hll = HyperLogLog(precision=14)   # 2^14 = 16384 registers
for i in range(1_000_000):
    hll.add(i)
print(hll.estimate())              # ~1,000,000

# Merge another counter with the same precision
other = HyperLogLog(precision=14)
other.update(["a", "b", "c"])
hll.merge(other)

# Independent copy and reset
dup = hll.copy()
hll.reset()
```

Exported names:

- `HyperLogLog` — the only public class.
  - `HyperLogLog(precision=14)` — `precision` must be an int in `[4, 16]`.
  - `.add(value)` — record one element. Any hashable Python object works.
  - `.update(iterable)` — record many elements.
  - `.estimate()` — return the current distinct-count estimate as a `float`.
  - `.merge(other)` — in-place merge; requires equal `precision`.
  - `.copy()` — return an independent copy.
  - `.reset()` — clear all registers.
  - `.precision`, `.num_registers` — read-only construction parameters.
  - `len(hll)` — integer estimate (convenience).

## Why this exists

Counting distinct elements exactly in a stream requires memory proportional to
the number of distinct values seen. HyperLogLog trades a small, fixed amount
of memory (a few kilobytes) for a modest relative error — roughly
`1.04 / sqrt(m)` where `m = 2^precision`. At `precision=14` that is about 1%.

The trade-off: the count is an estimate, not an exact figure. If you need an
exact answer or need to enumerate the distinct values themselves, this is the
wrong tool.

## Edge cases and decisions

- **Determinism.** Python's built-in `hash()` is randomized per process for
  strings and other hashable types, so it cannot be used for a stable sketch.
  This library hashes by finalizing the built-in `hash()` through splitmix64,
  which keeps determinism across processes while still mixing bits well.

- **Merge requires equal precision.** Merging counters of different precisions
  would demand register coalescing, which is easy to get wrong. Mismatched
  merges raise `ValueError` rather than guessing.

- **Unhashable inputs.** `add` accepts any hashable object; unhashable ones
  (lists, dicts, sets) raise `TypeError`, propagated from the built-in
  `hash()`.

- **Empty counter.** `estimate()` on a counter with no observations returns
  `0.0` exactly, via the small-range correction.
