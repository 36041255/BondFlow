import unittest
from types import SimpleNamespace

import torch

from BondFlow.models.guidance import SoftBondCountGuidance, TypeAwareSoftBondCountGuidance


class SoftBondCountOneOfNTests(unittest.TestCase):
    def make_guidance(self, **overrides):
        values = {"mode": "one_of_N", "weight": 0.1, "allowed_N": [1, 2]}
        values.update(overrides)
        return SoftBondCountGuidance(SimpleNamespace(**values))

    def test_one_of_n_has_wells_only_at_allowed_counts(self):
        guidance = self.make_guidance()

        energy, _ = guidance._count_energy(torch.tensor([1.0, 1.5, 2.0]))

        self.assertTrue(torch.allclose(energy, torch.tensor([0.0, 0.25, 0.0])))

    def test_one_of_n_requires_nonempty_finite_allowed_targets(self):
        for allowed in (None, [], [1, float("inf")]):
            with self.subTest(allowed=allowed), self.assertRaisesRegex(ValueError, "allowed_N"):
                self.make_guidance(allowed_N=allowed)


class TypeAwareSoftBondCountOneOfNTests(unittest.TestCase):
    def make_guidance(self, **overrides):
        type_config = SimpleNamespace(
            name="disulfide",
            mode="one_of_N",
            allowed_N=[1, 2],
            weight=1.0,
        )
        for key, value in overrides.items():
            setattr(type_config, key, value)
        return TypeAwareSoftBondCountGuidance(SimpleNamespace(types=[type_config]))

    def test_one_of_n_has_discrete_typed_count_wells(self):
        guidance = self.make_guidance()
        energy = TypeAwareSoftBondCountGuidance._one_of_count_energy(
            torch.tensor([1.0, 1.5, 2.0]),
            (1.0, 2.0),
            "mse",
            1e-8,
        )

        self.assertEqual(guidance.type_cfgs[0].allowed_N, (1.0, 2.0))
        self.assertTrue(torch.allclose(energy, torch.tensor([0.0, 0.25, 0.0])))

    def test_one_of_n_requires_nonempty_finite_allowed_targets(self):
        for allowed in (None, [], [1, float("inf")]):
            with self.subTest(allowed=allowed), self.assertRaisesRegex(ValueError, "allowed_N"):
                self.make_guidance(allowed_N=allowed)


if __name__ == "__main__":
    unittest.main()
