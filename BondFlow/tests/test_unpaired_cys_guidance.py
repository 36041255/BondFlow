import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from BondFlow.models.guidance import TypeAwareSoftBondCountGuidance
from rfdiff.chemical import aa2num


LINK_CSV = Path(__file__).resolve().parents[1] / "config" / "link.csv"


class UnpairedCysGuidanceTests(unittest.TestCase):
    def make_guidance(self, weight=1.0, type_name="disulfide", **overrides):
        values = dict(
            link_csv_path=str(LINK_CSV),
            unpaired_cys_weight=weight,
            suppress_adjacent_weight=0.0,
            types=[SimpleNamespace(name=type_name, target_N=0, weight=0.0)],
        )
        values.update(overrides)
        return TypeAwareSoftBondCountGuidance(SimpleNamespace(**values))

    def make_logits(self, guidance, sequence):
        logits = torch.full((1, len(sequence), guidance.num_aatypes), -30.0)
        for pos, residue in enumerate(sequence):
            logits[0, pos, aa2num[residue]] = 30.0
        return logits

    def energy(self, guidance, bonds, logits, **masks):
        return guidance._compute_energy(bonds, logits, None, None, None, **masks)

    def test_only_unpaired_cys_adds_energy(self):
        guidance = self.make_guidance(weight=0.5)
        logits = self.make_logits(guidance, ["CYS", "CYS", "CYS", "ALA"])
        bonds = torch.eye(4).unsqueeze(0)
        bonds[0, 0, 0] = bonds[0, 1, 1] = 0
        bonds[0, 0, 1] = bonds[0, 1, 0] = 1

        self.assertAlmostEqual(self.energy(guidance, bonds, logits).item(), 0.5)

    def test_no_cys_has_no_penalty(self):
        guidance = self.make_guidance()
        logits = self.make_logits(guidance, ["ALA", "GLY"])

        self.assertAlmostEqual(self.energy(guidance, torch.eye(2)[None], logits).item(), 0)

    def test_default_weight_preserves_existing_count_energy(self):
        guidance = self.make_guidance()
        del guidance.cfg.unpaired_cys_weight
        guidance = TypeAwareSoftBondCountGuidance(guidance.cfg)
        guidance.type_cfgs[0].weight = 2.0
        guidance.type_cfgs[0].target_N = 1.0
        logits = self.make_logits(guidance, ["CYS", "CYS", "CYS"])

        self.assertEqual(guidance.unpaired_cys_weight, 0.0)
        self.assertAlmostEqual(self.energy(guidance, torch.eye(3)[None], logits).item(), 2.0)

    def test_rejects_negative_or_nonfinite_weights(self):
        for weight in (-1.0, float("inf"), float("nan")):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "unpaired_cys_weight"):
                self.make_guidance(weight=weight)

    def test_region_and_mutable_masks_limit_penalized_positions(self):
        guidance = self.make_guidance()
        logits = self.make_logits(guidance, ["CYS"] * 4)
        region = torch.zeros((1, 4, 4), dtype=torch.bool)
        region[:, :2, :2] = True
        mutable = torch.tensor([[True, False, True, True]])

        energy = self.energy(guidance, torch.eye(4)[None], logits,
                             region_mask_2d=region, seq_mask=mutable)

        self.assertAlmostEqual(energy.item(), 1.0)

    def test_padded_residues_and_batches(self):
        guidance = self.make_guidance()
        logits = self.make_logits(guidance, ["CYS"] * 3).repeat(2, 1, 1)
        bonds = torch.eye(3)[None].repeat(2, 1, 1)
        valid = torch.tensor([[True, True, False], [True, False, False]])

        energy = guidance._compute_energy(bonds, logits, valid, None, None)

        self.assertAlmostEqual(energy.item(), 3.0)

    def test_empty_region_or_no_mutable_residues_has_zero_penalty(self):
        guidance = self.make_guidance()
        logits = self.make_logits(guidance, ["CYS", "CYS"])
        for masks in (dict(region_mask_2d=torch.zeros((1, 2, 2), dtype=torch.bool)),
                      dict(seq_mask=torch.zeros((1, 2), dtype=torch.bool))):
            with self.subTest(masks=masks):
                self.assertAlmostEqual(self.energy(guidance, torch.eye(2)[None], logits, **masks).item(), 0)

    def test_fixed_cys_can_still_be_a_partner(self):
        guidance = self.make_guidance()
        logits = self.make_logits(guidance, ["CYS", "CYS"])
        bonds = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

        energy = self.energy(guidance, bonds, logits, seq_mask=torch.tensor([[True, False]]))

        self.assertAlmostEqual(energy.item(), 0)

    def test_disallowed_pairs_do_not_count_as_paired(self):
        guidance = self.make_guidance()
        logits = self.make_logits(guidance, ["CYS", "CYS"])
        bonds = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        adjacent = torch.tensor([[[False, True], [True, False]]])

        energy = self.energy(guidance, bonds, logits, adjacent_mask_2d=adjacent)

        self.assertAlmostEqual(energy.item(), 2.0)

    def test_other_bond_types_do_not_add_cys_penalty(self):
        guidance = self.make_guidance(type_name="isopeptide")
        logits = self.make_logits(guidance, ["CYS", "CYS"])

        self.assertAlmostEqual(self.energy(guidance, torch.eye(2)[None], logits).item(), 0)

    def test_gradients_lower_unpaired_cys_and_encourage_pairing(self):
        guidance = self.make_guidance()
        logits = self.make_logits(guidance, ["CYS", "CYS", "CYS"])
        logits[:, :, aa2num["CYS"]] = 0
        logits[:, :, aa2num["ALA"]] = 0
        logits.requires_grad_()
        bonds = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]],
                             requires_grad=True)

        energy = self.energy(guidance, bonds, logits)
        grad_bonds, grad_logits = torch.autograd.grad(energy, (bonds, logits))

        self.assertTrue(torch.isfinite(grad_bonds).all())
        self.assertTrue(torch.isfinite(grad_logits).all())
        self.assertLess(grad_bonds[0, 0, 1].item(), 0)
        self.assertGreater(grad_logits[0, 2, aa2num["CYS"]].item(), 0)
        self.assertLess(grad_logits[0, 2, aa2num["ALA"]].item(), 0)

    def test_fixed_and_outside_positions_have_no_sequence_gradient(self):
        guidance = self.make_guidance()
        logits = torch.zeros((1, 4, guidance.num_aatypes), requires_grad=True)
        region = torch.zeros((1, 4, 4), dtype=torch.bool)
        region[:, :2, :2] = True
        energy = self.energy(guidance, torch.eye(4)[None], logits,
                             region_mask_2d=region,
                             seq_mask=torch.tensor([[True, False, True, True]]))

        gradient, = torch.autograd.grad(energy, (logits,))

        self.assertGreater(gradient[0, 0].abs().sum().item(), 0)
        self.assertEqual(gradient[0, 1:].abs().sum().item(), 0)

    def test_pre_model_passes_masks_and_reduces_mutable_cys_probability(self):
        guidance = self.make_guidance(stage="both", bond_step=0.0, seq_step=0.1,
                                      n_steps=1, region="A|A")
        logits = torch.zeros((1, 3, guidance.num_aatypes))
        before = logits.softmax(-1)[0, 0, aa2num["CYS"]].item()

        result = guidance.pre_model(
            {"bond_mat_pred": torch.eye(3)[None], "logits": logits.clone()},
            masks={"seq_mask": torch.tensor([[True, False, True]])},
            full_pdb_idx=[("A", 1), ("A", 2), ("B", 1)],
        )

        self.assertLess(result["logits"].softmax(-1)[0, 0, aa2num["CYS"]].item(), before)
        self.assertTrue(torch.equal(result["logits"][:, 1:], logits[:, 1:]))

    def test_post_step_passes_masks_and_can_replace_mutable_unpaired_cys(self):
        guidance = self.make_guidance(stage="both", bond_step=0.0, seq_step=10000,
                                      n_steps=1, region="A|A")
        sequence = torch.full((1, 3), aa2num["CYS"], dtype=torch.long)

        result = guidance.post_step(
            {"ss_t_2": torch.eye(3)[None], "aatypes_t_2": sequence.clone()},
            masks={"seq_mask": torch.tensor([[True, False, True]])},
            full_pdb_idx=[("A", 1), ("A", 2), ("B", 1)],
        )

        self.assertNotEqual(result["aatypes_t_2"][0, 0].item(), aa2num["CYS"])
        self.assertTrue(torch.equal(result["aatypes_t_2"][:, 1:], sequence[:, 1:]))


if __name__ == "__main__":
    unittest.main()
