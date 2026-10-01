import math
import unittest

from hyperloglog_counter import HyperLogLog


class TestConstruction(unittest.TestCase):
    def test_default_precision(self):
        h = HyperLogLog()
        self.assertEqual(h.precision, 14)
        self.assertEqual(h.num_registers, 1 << 14)

    def test_explicit_precision(self):
        h = HyperLogLog(precision=10)
        self.assertEqual(h.precision, 10)
        self.assertEqual(h.num_registers, 1024)

    def test_precision_bounds(self):
        with self.assertRaises(ValueError):
            HyperLogLog(precision=3)
        with self.assertRaises(ValueError):
            HyperLogLog(precision=17)

    def test_precision_type(self):
        with self.assertRaises(TypeError):
            HyperLogLog(precision=14.0)  # type: ignore[arg-type]


class TestEstimation(unittest.TestCase):
    def test_empty_estimate_is_zero(self):
        # All registers are zero; small-range correction yields m*log(m/m)=0.
        h = HyperLogLog(precision=12)
        self.assertEqual(h.estimate(), 0.0)
        self.assertEqual(len(h), 0)

    def test_single_element(self):
        h = HyperLogLog(precision=12)
        h.add("only-one")
        # At least one register is set; small-range correction applies and the
        # estimate is positive but small relative to 2^12.
        e = h.estimate()
        self.assertGreater(e, 0.0)
        self.assertLess(e, 100.0)

    def test_distinct_count_is_in_expected_range(self):
        # Deterministic count test. p=12 gives 4096 registers; for n=10000
        # distinct items the relative standard error is ~1.04/sqrt(4096) ~=
        # 1.6%. We use a generous 15% band: well outside the noise floor while
        # still catching gross regressions (e.g. a missing correction).
        h = HyperLogLog(precision=12)
        n = 10000
        for i in range(n):
            h.add(i)
        e = h.estimate()
        self.assertGreater(e, n * 0.85)
        self.assertLess(e, n * 1.15)

    def test_duplicates_do_not_increase_estimate(self):
        h1 = HyperLogLog(precision=10)
        h2 = HyperLogLog(precision=10)
        for i in range(5000):
            h1.add(i)
        for _ in range(100):
            for i in range(5000):
                h2.add(i)
        # The two estimates should be very close. We compare log-ratios to
        # avoid equality on floats.
        r = math.log(h1.estimate()) - math.log(h2.estimate())
        self.assertLess(abs(r), 0.2)

    def test_add_handles_unhashable_via_type_error(self):
        # We rely on the built-in hash(); unhashable objects raise TypeError,
        # which we let propagate rather than swallow.
        h = HyperLogLog()
        with self.assertRaises(TypeError):
            h.add([1, 2, 3])  # type: ignore[arg-type]

    def test_update_empty_iterable(self):
        h = HyperLogLog(precision=8)
        h.update([])
        self.assertEqual(h.estimate(), 0.0)

    def test_update_equivalent_to_repeated_add(self):
        h1 = HyperLogLog(precision=10)
        h2 = HyperLogLog(precision=10)
        items = list(range(2000))
        for x in items:
            h1.add(x)
        h2.update(items)
        self.assertEqual(list(h1._registers), list(h2._registers))


class TestDeterminism(unittest.TestCase):
    def test_same_element_same_register_and_rho(self):
        # The same value must always produce the same internal state, so two
        # fresh counters fed the same single element must be byte-identical.
        h1 = HyperLogLog(precision=12)
        h2 = HyperLogLog(precision=12)
        h1.add("deterministic-key")
        h2.add("deterministic-key")
        self.assertEqual(bytes(h1._registers), bytes(h2._registers))

    def test_estimate_is_deterministic_across_instances(self):
        h1 = HyperLogLog(precision=12)
        h2 = HyperLogLog(precision=12)
        for i in range(3000):
            h1.add(i)
            h2.add(i)
        self.assertEqual(h1.estimate(), h2.estimate())


class TestMerge(unittest.TestCase):
    def test_merge_disjoint_sets(self):
        h1 = HyperLogLog(precision=12)
        h2 = HyperLogLog(precision=12)
        for i in range(5000):
            h1.add(i)
        for i in range(5000, 10000):
            h2.add(i)
        h1.merge(h2)
        e = h1.estimate()
        self.assertGreater(e, 10000 * 0.85)
        self.assertLess(e, 10000 * 1.15)

    def test_merge_overlapping_sets(self):
        h1 = HyperLogLog(precision=12)
        h2 = HyperLogLog(precision=12)
        for i in range(5000):
            h1.add(i)
        for i in range(2500, 7500):
            h2.add(i)
        h1.merge(h2)
        # Union is {0..7499} -> 7500 distinct.
        e = h1.estimate()
        self.assertGreater(e, 7500 * 0.85)
        self.assertLess(e, 7500 * 1.15)

    def test_merge_precision_mismatch(self):
        h1 = HyperLogLog(precision=10)
        h2 = HyperLogLog(precision=12)
        with self.assertRaises(ValueError):
            h1.merge(h2)

    def test_merge_type_error(self):
        h = HyperLogLog()
        with self.assertRaises(TypeError):
            h.merge("not-an-hll")  # type: ignore[arg-type]

    def test_merge_does_not_mutate_source(self):
        h1 = HyperLogLog(precision=10)
        h2 = HyperLogLog(precision=10)
        for i in range(1000):
            h1.add(i)
        for i in range(2000):
            h2.add(i)
        before = bytes(h2._registers)
        h1.merge(h2)
        self.assertEqual(bytes(h2._registers), before)


class TestCopyAndReset(unittest.TestCase):
    def test_copy_is_independent(self):
        h = HyperLogLog(precision=10)
        for i in range(1000):
            h.add(i)
        c = h.copy()
        for i in range(1000, 2000):
            h.add(i)
        # c must not reflect the post-copy additions.
        self.assertNotEqual(bytes(h._registers), bytes(c._registers))

    def test_copy_preserves_estimate(self):
        h = HyperLogLog(precision=12)
        for i in range(5000):
            h.add(i)
        c = h.copy()
        self.assertEqual(h.estimate(), c.estimate())

    def test_reset_clears_state(self):
        h = HyperLogLog(precision=10)
        for i in range(1000):
            h.add(i)
        h.reset()
        self.assertEqual(h.estimate(), 0.0)
        # And is usable again.
        h.add("after-reset")
        self.assertGreater(h.estimate(), 0.0)


class TestHashCoverage(unittest.TestCase):
    def test_registers_are_populated_not_all_in_one(self):
        # If register selection were broken (e.g. always 0), all updates would
        # hit a single register. Sanity-check that at least 10% of registers
        # receive a value for a moderately-sized input.
        h = HyperLogLog(precision=12)
        for i in range(20000):
            h.add(i)
        nonzero = sum(1 for r in h._registers if r > 0)
        self.assertGreater(nonzero, h.num_registers * 0.10)

    def test_string_and_int_inputs_both_supported(self):
        # The library hashes arbitrary Python objects; strings and ints must
        # both work and yield different register distributions.
        hs = HyperLogLog(precision=10)
        hi = HyperLogLog(precision=10)
        for i in range(2000):
            hs.add(f"item-{i}")
            hi.add(i)
        self.assertNotEqual(bytes(hs._registers), bytes(hi._registers))


if __name__ == "__main__":
    unittest.main()
