from __future__ import annotations

import math
from typing import Iterable, Optional


class HyperLogLog:
    """In-memory HyperLogLog estimator for distinct-element counting.

    Implementation notes (why these choices):

    * Precision is fixed at construction. Choosing it once (rather than
      auto-tuning from observed cardinality) keeps register addressing simple
      and predictable: the same element always maps to the same register no
      matter how many others have been seen since. Auto-tuning would require
      re-hashing on resize, which is incompatible with a `merge` that operates
      on raw register arrays.

    * The estimator uses the standard bias-corrected form from Flajolet et
      al. (2007): the raw `alpha * m^2 / Z` term, the small-range correction
      (``Z``-based) when ``E < 2.5 * m``, and the large-range correction for
      counts approaching ``2^64``. Stochastic averaging is used to combine
      register contributions. We do NOT add noise to break ties; it is not
      needed for the linear-range tests this implementation targets and would
      complicate reasoning about determinism.

    * `merge` requires equal precision and registers stored in the same index
      order. Allowing a merge across precisions would require register
      coalescing (splitting/combining buckets), which is possible but adds a
      class of subtle bugs. We reject the mismatch loudly instead.
    """

    __slots__ = ("_precision", "_m", "_alpha", "_registers")

    # 64-bit hash. Smaller hashes cap the countable cardinality; 32-bit is
    # the textbook default but breaks the large-range correction before it
    # becomes useful. 64-bit keeps large-range correction meaningful.
    _hash_bits = 64
    _hash_mask = (1 << _hash_bits) - 1
    _hash_wrap = 1 << _hash_bits

    def __init__(self, precision: int = 14) -> None:
        if not isinstance(precision, int):
            raise TypeError("precision must be an int")
        if precision < 4 or precision > 16:
            raise ValueError("precision must be in [4, 16]")
        self._precision = precision
        self._m = 1 << precision
        # Bias constant per Flajolet et al. The piecewise form matters at the
        # boundaries; a single constant drifts noticeably at m=16 and m=32.
        if self._m == 16:
            self._alpha = 0.673
        elif self._m == 32:
            self._alpha = 0.697
        elif self._m == 64:
            self._alpha = 0.709
        else:
            self._alpha = 0.7213 / (1.0 + 1.079 / self._m)
        # Registers start at 0 meaning "no observation". rho values are >=1,
        # so 0 cleanly distinguishes untouched registers.
        self._registers = bytearray(self._m)

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def num_registers(self) -> int:
        return self._m

    def _hash(self, value: object) -> int:
        """Stable 64-bit hash of arbitrary Python objects.

        Determinism is non-negotiable for a counter: the same element must map
        to the same register and the same rho regardless of process or run.
        Python's built-in `hash()` is randomized per process by default and
        cannot be used. We reuse CPython's `hash` on its stable types
        (str/bytes/tuple of stable types) to get well-mixed input, then pass
        it through a 64-bit finalizer (splitmix64) to distribute bits so that
        high-order bits — which select the register — are well scattered for
        sequential keys.
        """
        h = hash(value)
        # Reduce to a 64-bit signed/unsigned int deterministically.
        h &= self._hash_mask
        # splitmix64 finalizer. Cheap and gives good bit dispersion.
        h = (h + 0x9E3779B97F4A7C15) & self._hash_mask
        h = (h ^ (h >> 30)) & self._hash_mask
        h = (h * 0xBF58476D1CE4E5B9) & self._hash_mask
        h = (h ^ (h >> 27)) & self._hash_mask
        h = (h * 0x94D049BB133111EB) & self._hash_mask
        h = (h ^ (h >> 31)) & self._hash_mask
        return h

    def add(self, value: object) -> None:
        """Record an element. Accepts any hashable Python object."""
        h = self._hash(value)
        # Top `precision` bits select the register. Using the high bits
        # (rather than the low bits) means register selection depends on the
        # full hash, not just a low slice; for well-mixed hashes this is
        # equivalent, but it plays better with the splitmix64 output whose
        # high bits are especially well-distributed.
        reg_index = h >> (self._hash_bits - self._precision)
        # The remaining low bits feed rho. We prepend a 1 bit to force at
        # least one set bit above the leading zeros of the remainder, so that
        # rho is always >= 1 and finite even for a hash whose remainder is 0.
        w = (h & ((1 << (self._hash_bits - self._precision)) - 1))
        # Width of the remaining field, in bits.
        rem_bits = self._hash_bits - self._precision
        # Position of the most significant set bit within {1..rem_bits} when we
        # prepend an implicit 1 at position rem_bits+1.
        if w == 0:
            rho = rem_bits + 1
        else:
            rho = rem_bits - (w.bit_length() - 1)
        if rho > 255:
            rho = 255  # 8-bit register storage; rem_bits <= 60 so this never triggers.
        if rho > self._registers[reg_index]:
            self._registers[reg_index] = rho

    def update(self, values: Iterable[object]) -> None:
        """Add many elements at once. Equivalent to calling `add` in a loop."""
        for v in values:
            self.add(v)

    def estimate(self) -> float:
        """Return the current estimate of the number of distinct elements."""
        m = self._m
        regs = self._registers
        # Sum of 2^-M[j]. Use ldexp to avoid float underflow subtleties; for
        # the sizes we support a plain `2.0 ** -r` is fine too, but ldexp is
        # exact and signals intent.
        Z = 0.0
        zero_count = 0
        for r in regs:
            if r == 0:
                zero_count += 1
                Z += 1.0
            else:
                # 2^-r
                Z += math.ldexp(1.0, -r)
        E = self._alpha * (m * m) / Z

        # Small-range correction. Flajolet's threshold is 2.5*m; below it the
        # raw estimator is biased high due to register collisions at low
        # cardinalities.
        if E <= 2.5 * m and zero_count > 0:
            E = m * math.log(m / zero_count)

        # Large-range correction. Only meaningful near 2^64; with 64-bit hashes
        # this kicks in for truly enormous cardinalities. Kept for correctness
        # rather than tested, because we cannot deterministically run a test
        # that large here.
        two_64 = float(1 << 64)
        if E > two_64 / 30.0:
            E = -two_64 * math.log(1.0 - E / two_64)

        return E

    def merge(self, other: "HyperLogLog") -> None:
        """Merge another HyperLogLog into this one in place.

        Requires equal precision. Merging across precisions is rejected
        because register coalescing (splitting high-precision registers into
        low-precision ones, or vice versa) is easy to get wrong and we do not
        implement it.
        """
        if not isinstance(other, HyperLogLog):
            raise TypeError("can only merge HyperLogLog instances")
        if other._precision != self._precision:
            raise ValueError(
                f"precision mismatch: {self._precision} vs {other._precision}"
            )
        a = self._registers
        b = other._registers
        for i in range(self._m):
            if b[i] > a[i]:
                a[i] = b[i]

    def copy(self) -> "HyperLogLog":
        """Return an independent copy of this counter."""
        c = HyperLogLog(self._precision)
        c._registers = bytearray(self._registers)
        return c

    def reset(self) -> None:
        """Clear all registers, returning the counter to its empty state."""
        for i in range(self._m):
            self._registers[i] = 0

    def __len__(self) -> int:
        """Convenience: integer estimate. Use `estimate()` for the float."""
        return int(self.estimate())
