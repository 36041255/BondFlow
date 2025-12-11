import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import MMCIFParser, ShrakeRupley, PDBExceptions

from aa_properties import aa_one, MAX_ASA, STANDARD_AA_THREE
from cif_utils import load_mmcif_dict, build_atom_index, parse_struct_conns, parse_secondary_structure_sets, get_chain_boundaries, compute_dssp_secondary_structure_sets
from geometry import distance, angle, dihedral
from utils import classify_endpoint, classify_pair, label_residue_property
from link_rules import LinkRuleBook


def list_cif_files(cif_dir: str) -> List[str]:
	paths: List[str] = []
	for root, _, files in os.walk(cif_dir):
		for f in files:
			if f.lower().endswith(".cif") or f.lower().endswith(".mmcif"):
				paths.append(os.path.join(root, f))
	return sorted(paths)


def build_structure_for_sasa(path: str):
	parser = MMCIFParser(QUIET=True)
	try:
		struct = parser.get_structure(os.path.basename(path), path)
	except Exception:
		return None
	try:
		sr = ShrakeRupley()
		sr.compute(struct, level="R")
	except Exception:
		pass
	return struct


def residue_rSASA(residue) -> Optional[float]:
	asa = residue.xtra.get("EXP_NASA")
	if asa is None:
		return None
	resname = residue.get_resname().upper()
	one = aa_one(resname)
	max_asa = MAX_ASA.get(one)
	if not max_asa:
		return None
	return float(asa) / float(max_asa)


def build_residue_sasa_map(struct) -> Dict[Tuple[str, int], float]:
	if struct is None:
		return {}
	m: Dict[Tuple[str, int], float] = {}
	for model in struct:
		for chain in model:
			chain_id = chain.id
			for res in chain:
				if res.id[0] != " ":
					continue
				seq = int(res.id[1])
				val = residue_rSASA(res)
				if val is not None:
					m[(chain_id, seq)] = float(val)
		break
	return m


def ensure_dirs(out_dir: str) -> Dict[str, str]:
	plots = os.path.join(out_dir, "plots")
	csv = os.path.join(out_dir, "csv")
	logs = os.path.join(out_dir, "logs")
	os.makedirs(plots, exist_ok=True)
	os.makedirs(csv, exist_ok=True)
	os.makedirs(logs, exist_ok=True)
	return {"plots": plots, "csv": csv, "logs": logs}


def replot_from_csv(csv_dir: str, plots_dir: Optional[str] = None, seq_span_ybreak: Optional[Tuple[float, float]] = None) -> None:
	"""
	Regenerate plots using existing CSV files only (no CIF/stat recomputation).
	"""
	csv_dir = os.path.abspath(csv_dir)
	if plots_dir is None:
		base = os.path.dirname(csv_dir)
		plots_dir = os.path.join(base, "plots")
	plots_dir = os.path.abspath(plots_dir)
	os.makedirs(plots_dir, exist_ok=True)

	from plotting import (
		barplot_counts,
		histplot,
		countplot,
		overlay_hist_by_hue,
		pieplot_from_counts,
		overlay_kde_by_hue,
		combined_kde_groups_by_chem,
		kdeplot_simple,
		hist_density_with_ybreak,
	)

	def _csv(name: str) -> str:
		return os.path.join(csv_dir, name)

	df_links = pd.read_csv(_csv("links.csv")) if os.path.exists(_csv("links.csv")) else pd.DataFrame()
	df_sum = pd.read_csv(_csv("per_entry_summary.csv")) if os.path.exists(_csv("per_entry_summary.csv")) else pd.DataFrame()

	if df_links.empty and (df_sum is None or df_sum.empty):
		print(f"No usable CSV files found in: {csv_dir}")
		return

	chem_palette = {"disulfide": "#FF7F0E", "amide": "#1F77B4", "lactone": "#D62728"}

	# Link type bar (log-scale)
	if os.path.exists(_csv("link_type_counts.csv")):
		link_type_counts = pd.read_csv(_csv("link_type_counts.csv"))
		if not link_type_counts.empty:
			barplot_counts(
				link_type_counts,
				"link_type",
				"count",
				"Positional link type counts (log y)",
				os.path.join(plots_dir, "link_type_bar.png"),
				log_y=True,
				rotation=0,
			)

	# Chemical type pie
	if os.path.exists(_csv("chem_type_counts.csv")):
		chem_counts = pd.read_csv(_csv("chem_type_counts.csv"))
		if not chem_counts.empty:
			pieplot_from_counts(
				chem_counts,
				"chem_type",
				"count",
				"Chemical link type proportion",
				os.path.join(plots_dir, "chem_type_pie.png"),
			)

	# Geometry distributions by chemical type (use only selected chem types)
	if not df_links.empty:
		df_geom = df_links[df_links["chem_type"].isin(["disulfide", "lactone", "amide"])].copy()

		if df_geom["bond_length"].notna().any():
			overlay_kde_by_hue(
				df_geom,
				"bond_length",
				"chem_type",
				"Bond length by chemical type (KDE)",
				os.path.join(plots_dir, "bond_length_by_chem.png"),
				xlim=None,
				bw_adjust=0.8,
				palette=chem_palette,
				clip_quantiles=(0.01, 0.99),
				fill_alpha=0.05,
				ax_label="Bond length (Å)",
			)

		# Angles and dihedrals (grouped by chem type and anchor group)
		df_angles = df_geom
		if df_angles[["angle_i", "angle_j"]].notna().any().any():
			angle_label_map = {
				"amide": {"i-anchor": "Cx-Cx-NZ", "j-anchor": "Cx-NZ-Cx"},
				"lactone": {"i-anchor": "Cx-Ox-Cx", "j-anchor": "Ox-Cx-Cx"},
				"disulfide": {"i-anchor": "Cx-SG-Sx"},
			}
			combined_kde_groups_by_chem(
				df_angles,
				["angle_i", "angle_j"],
				["i-anchor", "j-anchor"],
				"chem_type",
				"Bond angle by chemical type (KDE)",
				os.path.join(plots_dir, "angles_by_chem.png"),
				xlim=None,
				label_map=angle_label_map,
				skip_second_for_disulfide=True,
				clip_quantiles=(0.01, 0.99),
				palette=chem_palette,
				fill_alpha=0.05,
				ax_label="Bond angle (deg)",
			)

		if df_angles[["dihedral_1", "dihedral_2"]].notna().any().any():
			dihedral_label_map = {
				"amide": {"group1": "Cx-Cx-NZ-Cx", "group2": "Ox=Cx-NZ-Cx"},
				"lactone": {"group1": "Cx-Ox-Cx-Cx", "group2": "Ox=Cx-Cx-Cx"},
				"disulfide": {"group1": "Cx-SG-SG-Cx"},
			}
			combined_kde_groups_by_chem(
				df_angles,
				["dihedral_1", "dihedral_2"],
				["group1", "group2"],
				"chem_type",
				"Dihedral by chemical type (KDE)",
				os.path.join(plots_dir, "dihedrals_by_chem.png"),
				xlim=(-180, 180),
				label_map=dihedral_label_map,
				skip_second_for_disulfide=True,
				clip_quantiles=(0.01, 0.99),
				palette=chem_palette,
				bw_adjust=0.1,
				fill_alpha=0.05,
				ax_label="Dihedral (deg)",
			)

		# AA pair top20
		if os.path.exists(_csv("aa_pair_top20.csv")):
			aa_counts = pd.read_csv(_csv("aa_pair_top20.csv"))
			if not aa_counts.empty:
				barplot_counts(
					aa_counts,
					"aa_pair",
					"count",
					"Top AA pair (disulfide/lactone/amide)",
					os.path.join(plots_dir, "aa_pair_top20.png"),
				)

		# Sequence span KDE (same-chain, overall)
		if "same_chain" in df_links.columns and "sequence_span" in df_links.columns:
			df_span_all = df_links[df_links["same_chain"] & df_links["sequence_span"].notna()].copy()
			seq_span_series = df_span_all["sequence_span"]
			if not seq_span_series.empty:
				hist_density_with_ybreak(
					seq_span_series,
					"Sequence span (same chain) density",
					os.path.join(plots_dir, "sequence_span_kde.png"),
					xlabel="Residue index separation",
					xlim=(0, min(300, float(seq_span_series.max()))),
					clip_quantiles=(0.01, 0.99),
					alpha=0.45,
					color="#1F77B4",
					y_break=(0.02, 0.23),
					y_upper=0.245,
				)

				# Sequence span KDE by chemical type (same-chain only)
				if "chem_type" in df_span_all.columns:
					# Optionally focus on the main chemical types of interest; keep others as-is
					df_span_chem = df_span_all[df_span_all["chem_type"].notna()].copy()
					if not df_span_chem.empty:
						overlay_kde_by_hue(
							df_span_chem,
							"sequence_span",
							"chem_type",
							"Sequence span (same chain) by chemical type (KDE)",
							os.path.join(plots_dir, "sequence_span_by_chem.png"),
							xlim=(0, min(300, float(seq_span_series.max()))),
							bw_adjust=0.8,
							palette=None,
							clip_quantiles=(0.01, 0.99),
							fill_alpha=0.35,
							ax_label="Residue index separation",
						)

		# Endpoint amino acid distribution pie
		if os.path.exists(_csv("amino_acid_counts.csv")):
			aa_dist = pd.read_csv(_csv("amino_acid_counts.csv"))
			if not aa_dist.empty:
				pieplot_from_counts(
					aa_dist,
					"res_three",
					"count",
					"Amino-acid distribution (endpoints)",
					os.path.join(plots_dir, "amino_acid_pie.png"),
				)

		# Overall amino-acid distribution pie
		if os.path.exists(_csv("amino_acid_overall_counts.csv")):
			aa_all = pd.read_csv(_csv("amino_acid_overall_counts.csv"))
			if not aa_all.empty:
				pieplot_from_counts(
					aa_all,
					"res_three",
					"count",
					"Amino-acid distribution (overall)",
					os.path.join(plots_dir, "amino_acid_overall_pie.png"),
				)

		# Residue property counts
		if os.path.exists(_csv("residue_property_counts.csv")):
			prop_counts = pd.read_csv(_csv("residue_property_counts.csv"))
			if not prop_counts.empty:
				pieplot_from_counts(
					prop_counts,
					"property",
					"count",
					"Residue property proportion",
					os.path.join(plots_dir, "residue_property_pie.png"),
				)

		# Secondary structure counts
		if os.path.exists(_csv("secondary_structure_counts.csv")):
			ss_counts = pd.read_csv(_csv("secondary_structure_counts.csv"))
			if not ss_counts.empty:
				pieplot_from_counts(
					ss_counts,
					"ss",
					"count",
					"Secondary structure proportion (helix/sheet/coil)",
					os.path.join(plots_dir, "secondary_structure_pie.png"),
				)

		# Exposure counts
		if os.path.exists(_csv("exposure_counts.csv")):
			exp_counts = pd.read_csv(_csv("exposure_counts.csv"))
			if not exp_counts.empty:
				pieplot_from_counts(
					exp_counts,
					"exposure",
					"count",
					"Exposure proportion (buried/exposed)",
					os.path.join(plots_dir, "exposure_pie.png"),
				)

	# Per-entry summary histograms
	if df_sum is not None and not df_sum.empty:
		if "num_links" in df_sum.columns:
			histplot(
				df_sum,
				"num_links",
				"Links per protein",
				os.path.join(plots_dir, "links_per_protein.png"),
				bins=40,
				log=True,
			)
		if "norm_links" in df_sum.columns:
			histplot(
				df_sum,
				"norm_links",
				"Normalized links (N/L^2)",
				os.path.join(plots_dir, "norm_links_per_protein.png"),
				bins=40,
				log=True,
			)

	print(f"Done. Plots written to: {plots_dir}")


def is_backbone_peptide(p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
	"""
	Exclude canonical backbone peptide bonds within the same chain:
	C(i) – N(i+1) or N(i) – C(i+1).
	"""
	if p1["chain"] != p2["chain"]:
		return False
	a1 = (p1["atom"] or "").upper()
	a2 = (p2["atom"] or "").upper()
	s1 = int(p1["seq"])
	s2 = int(p2["seq"])
	if a1 == "C" and a2 == "N" and (s2 - s1) == 1:
		return True
	if a1 == "N" and a2 == "C" and (s1 - s2) == 1:
		return True
	return False


def main():
	parser = argparse.ArgumentParser(description="CycLinkDB dataset statistics")
	parser.add_argument("--cif_dir", type=str, required=False, help="Directory containing mmCIF files")
	parser.add_argument("--out_dir", type=str, required=False, help="Output directory for CSVs and plots")
	parser.add_argument("--csv_dir", type=str, required=False, help="If set, skip CIF processing and regenerate plots from this CSV directory only")
	parser.add_argument("--plots_dir", type=str, required=False, help="Output directory for plots in CSV-only mode (default: sibling 'plots' next to csv_dir)")
	parser.add_argument("--n_workers", type=int, default=1, help="Reserved (not used)")
	parser.add_argument("--sasa_threshold", type=float, default=0.1, help="Relative SASA threshold to call buried/exposed")
	parser.add_argument("--skip_sasa", action="store_true", help="Skip SASA computation to speed up")
	parser.add_argument("--skip_secondary", action="store_true", help="Skip parsing/DSSP secondary structure annotation")
	parser.add_argument("--link_csv_path", type=str, default="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv", help="Path to link.csv to guide geometry anchors and types")
	args = parser.parse_args()
	# CSV-only mode: regenerate plots from existing CSV folder, skip CIF/stat computation
	if getattr(args, "csv_dir", None):
		replot_from_csv(args.csv_dir, getattr(args, "plots_dir", None))
		return
	# Normal mode requires cif_dir and out_dir
	if not args.cif_dir or not args.out_dir:
		parser.error("arguments --cif_dir and --out_dir are required unless --csv_dir is provided")
	paths = list_cif_files(args.cif_dir)
	os.makedirs(args.out_dir, exist_ok=True)
	dirs = ensure_dirs(args.out_dir)

	# Resolve default link.csv path if not provided
	if not args.link_csv_path:
		here = os.path.dirname(os.path.abspath(__file__))
		repo_root = os.path.dirname(here)
		default_csv = os.path.join(repo_root, "BondFlow", "config", "link.csv")
		args.link_csv_path = default_csv if os.path.exists(default_csv) else ""
	rulebook = LinkRuleBook(args.link_csv_path) if args.link_csv_path else None

	records: List[Dict[str, Any]] = []
	dup_keys: Set[Tuple] = set()
	per_entry_counts: Dict[str, int] = defaultdict(int)
	per_entry_lengths: Dict[str, int] = {}
	aa_overall_counts: Counter = Counter()

	for path in tqdm(paths, desc="Processing CIF"):
		entry_id = os.path.splitext(os.path.basename(path))[0]
		try:
			mmcif = load_mmcif_dict(path)
		except Exception:
			continue
		conns = parse_struct_conns(mmcif)
		# If no covalent links in this entry, skip heavy steps
		if not conns:
			# still record length for coverage and amino-acid overall distribution
			_, residue_name_tmp, chain_sequences_tmp = build_atom_index(mmcif)
			per_entry_lengths[entry_id] = sum(len(s) for s in chain_sequences_tmp.values())
			# overall AA counts (three-letter)
			for (_, _), res3 in residue_name_tmp.items():
				if res3:
					aa_overall_counts[res3] += 1
			continue
		# Build indices only when needed
		atom_index, residue_name, chain_sequences = build_atom_index(mmcif)
		chain_bounds = get_chain_boundaries(mmcif)
		per_entry_lengths[entry_id] = sum(len(s) for s in chain_sequences.values())
		if args.skip_secondary:
			helix_set, sheet_set = set(), set()
		else:
			helix_set, sheet_set = parse_secondary_structure_sets(mmcif)
			# Fallback to DSSP if CIF lacks secondary structure annotations
			if len(helix_set) == 0 and len(sheet_set) == 0:
				try:
					h2, s2 = compute_dssp_secondary_structure_sets(path)
					if h2 or s2:
						helix_set, sheet_set = h2, s2
				except Exception:
					# DSSP not available or failed; keep empty sets
					pass
		if args.skip_sasa:
			rSASA_map = {}
		else:
			struct = build_structure_for_sasa(path)
			rSASA_map = build_residue_sasa_map(struct)
		# overall AA counts for this entry
		for (_, _), res3 in residue_name.items():
			if res3:
				aa_overall_counts[res3] += 1
		# Build a global sequence key for dedup: concatenate chains in sorted order
		sequence_key = "|".join([f"{ch}:{chain_sequences[ch]}" for ch in sorted(chain_sequences.keys())])
		for row in conns:
			p1 = row["p1"]
			p2 = row["p2"]
			# Exclude canonical backbone peptide bonds
			if is_backbone_peptide(p1, p2):
				continue
			# Only consider standard 20 aa at both sides
			if p1["comp"] not in STANDARD_AA_THREE or p2["comp"] not in STANDARD_AA_THREE:
				continue
			# Locate coordinates
			coord1 = atom_index.get((p1["chain"], p1["seq"], p1["atom"]))
			coord2 = atom_index.get((p2["chain"], p2["seq"], p2["atom"]))
			if coord1 is None or coord2 is None:
				continue
			# Dedup identical sequence + identical link endpoints (order independent)
			end_a = (p1["chain"], int(p1["seq"]), p1["atom"])
			end_b = (p2["chain"], int(p2["seq"]), p2["atom"])
			link_key = (sequence_key, tuple(sorted([end_a, end_b])))
			if link_key in dup_keys:
				continue
			dup_keys.add(link_key)
			# Classification (positional)
			t1 = classify_endpoint(p1["chain"], int(p1["seq"]), p1["atom"], chain_bounds)
			t2 = classify_endpoint(p2["chain"], int(p2["seq"]), p2["atom"], chain_bounds)
			link_type = classify_pair(t1, t2)
			# Chemical type classification
			aa1_three = residue_name.get((p1["chain"], int(p1["seq"])))
			aa2_three = residue_name.get((p2["chain"], int(p2["seq"])))
			chem_type = LinkRuleBook.classify_chem_type(aa1_three or "", aa2_three or "", p1["atom"], p2["atom"], row.get("type", ""))
			# Compute geometry using rulebook anchors if available; else fallback to CA-based
			bond_len = distance(coord1, coord2)
			bond_ang = None
			torsion = None
			angle_i_val = None
			angle_j_val = None
			dihedral1_val = None
			dihedral2_val = None
			if rulebook is not None and aa1_three and aa2_three:
				rule, swapped = rulebook.find_rule(aa1_three, aa2_three, p1["atom"], p2["atom"])
				if rule:
					# Determine anchor names respecting orientation
					ai = rule.get("angle_i_anchor")
					aj = rule.get("angle_j_anchor")
					d1_i = rule.get("dihedral_1_anchor_i")
					d1_j = rule.get("dihedral_1_anchor_j")
					d2_i = rule.get("dihedral_2_anchor_i")
					d2_j = rule.get("dihedral_2_anchor_j")
					# If swapped, i<->j anchors swap residues
					if swapped:
						ai, aj = aj, ai
						d1_i, d1_j = d1_j, d1_i
						d2_i, d2_j = d2_j, d2_i
					# Compute angle_i (anchor on i side) and angle_j (anchor on j side)
					try:
						p_ai = atom_index.get((p1["chain"], int(p1["seq"]), ai)) if ai else None
						if p_ai is not None:
							angle_i_val = angle(p_ai, coord1, coord2)
					except Exception:
						angle_i_val = angle_i_val
					try:
						p_aj = atom_index.get((p2["chain"], int(p2["seq"]), aj)) if aj else None
						if p_aj is not None:
							angle_j_val = angle(coord1, coord2, p_aj)
					except Exception:
						angle_j_val = angle_j_val
					# Legacy single-angle fallback for compatibility
					if bond_ang is None:
						bond_ang = angle_i_val if angle_i_val is not None else angle_j_val
					# Compute dihedral group 1 and 2 (if anchors exist)
					try:
						if d1_i and d1_j:
							p_d1i = atom_index.get((p1["chain"], int(p1["seq"]), d1_i))
							p_d1j = atom_index.get((p2["chain"], int(p2["seq"]), d1_j))
							if p_d1i is not None and p_d1j is not None:
								dihedral1_val = dihedral(p_d1i, coord1, coord2, p_d1j)
					except Exception:
						dihedral1_val = dihedral1_val
					try:
						if d2_i and d2_j:
							p_d2i = atom_index.get((p1["chain"], int(p1["seq"]), d2_i))
							p_d2j = atom_index.get((p2["chain"], int(p2["seq"]), d2_j))
							if p_d2i is not None and p_d2j is not None:
								dihedral2_val = dihedral(p_d2i, coord1, coord2, p_d2j)
					except Exception:
						dihedral2_val = dihedral2_val
					# Legacy single-dihedral fallback for compatibility
					if torsion is None:
						torsion = dihedral1_val if dihedral1_val is not None else dihedral2_val
			# Secondary structure and SASA for residues
			if args.skip_secondary:
				ss1 = None
				ss2 = None
			else:
				ss1 = "coil"
				ss2 = "coil"
				if (p1["chain"], int(p1["seq"])) in helix_set:
					ss1 = "helix"
				elif (p1["chain"], int(p1["seq"])) in sheet_set:
					ss1 = "sheet"
				if (p2["chain"], int(p2["seq"])) in helix_set:
					ss2 = "helix"
				elif (p2["chain"], int(p2["seq"])) in sheet_set:
					ss2 = "sheet"
			rsasa1 = rSASA_map.get((p1["chain"], int(p1["seq"])))
			rsasa2 = rSASA_map.get((p2["chain"], int(p2["seq"])))
			exp1 = None if rsasa1 is None else ("exposed" if rsasa1 >= args.sasa_threshold else "buried")
			exp2 = None if rsasa2 is None else ("exposed" if rsasa2 >= args.sasa_threshold else "buried")
			# Sequence span if same chain
			seq_span = None
			if p1["chain"] == p2["chain"]:
				seq_span = abs(int(p1["seq"]) - int(p2["seq"]))
			aa1_one = aa_one(aa1_three) if aa1_three else "X"
			aa2_one = aa_one(aa2_three) if aa2_three else "X"
			records.append({
				"entry_id": entry_id,
				"link_id": row["id"],
				"conn_type_id": row["type"],
				"chem_type": chem_type,
				"angle_i": angle_i_val,
				"angle_j": angle_j_val,
				"dihedral_1": dihedral1_val,
				"dihedral_2": dihedral2_val,
				"chain1": p1["chain"],
				"seq1": int(p1["seq"]),
				"atom1": p1["atom"],
				"res1_three": aa1_three,
				"res1_one": aa1_one,
				"class1": t1,
				"ss1": ss1,
				"rSASA1": rsasa1,
				"exposure1": exp1,
				"chain2": p2["chain"],
				"seq2": int(p2["seq"]),
				"atom2": p2["atom"],
				"res2_three": aa2_three,
				"res2_one": aa2_one,
				"class2": t2,
				"ss2": ss2,
				"rSASA2": rsasa2,
				"exposure2": exp2,
				"link_type": link_type,
				"bond_length": bond_len,
				"bond_angle": bond_ang,
				"dihedral": torsion,
				"same_chain": p1["chain"] == p2["chain"],
				"sequence_span": seq_span,
			})
			per_entry_counts[entry_id] += 1

	# Save links table
	df = pd.DataFrame.from_records(records)
	df.to_csv(os.path.join(dirs["csv"], "links.csv"), index=False)

	# Per-entry summary
	sum_rows = []
	for entry, n_links in per_entry_counts.items():
		length = max(1, per_entry_lengths.get(entry, 1))
		sum_rows.append({
			"entry_id": entry,
			"num_links": n_links,
			"length": length,
			"norm_links": n_links / float(length * length),
		})
	df_sum = pd.DataFrame(sum_rows)
	df_sum.to_csv(os.path.join(dirs["csv"], "per_entry_summary.csv"), index=False)

	# Global counts
	num_entries = len(per_entry_lengths)
	num_with_links = df["entry_id"].nunique() if not df.empty else 0
	global_summary = {
		"num_entries": num_entries,
		"num_with_links": num_with_links,
		"coverage_pct": (100.0 * num_with_links / max(1, num_entries)),
		"num_links_total": int(df.shape[0]),
	}
	with open(os.path.join(dirs["csv"], "global_summary.json"), "w") as f:
		json.dump(global_summary, f, indent=2)

	# Link type distribution
	if not df.empty:
		link_type_counts = df["link_type"].value_counts().rename_axis("link_type").reset_index(name="count")
		link_type_counts["percent"] = 100.0 * link_type_counts["count"] / link_type_counts["count"].sum()
		link_type_counts.to_csv(os.path.join(dirs["csv"], "link_type_counts.csv"), index=False)
		# Chemical type distribution
		chem_counts = df["chem_type"].value_counts().rename_axis("chem_type").reset_index(name="count")
		chem_counts["percent"] = 100.0 * chem_counts["count"] / chem_counts["count"].sum()
		chem_counts.to_csv(os.path.join(dirs["csv"], "chem_type_counts.csv"), index=False)

		# Amino acid pair frequencies (top 20)
		def pair_label(a: str, b: str) -> str:
			return "-".join(sorted([a, b]))
		df_types = df[df["chem_type"].isin(["disulfide", "lactone", "amide"])].copy()
		df_types["aa_pair"] = df_types.apply(lambda r: pair_label(r["res1_three"] or "UNK", r["res2_three"] or "UNK"), axis=1)
		aa_counts = df_types["aa_pair"].value_counts().head(20).rename_axis("aa_pair").reset_index(name="count")
		aa_counts.to_csv(os.path.join(dirs["csv"], "aa_pair_top20.csv"), index=False)

		# Amino acid distribution (endpoints)
		aa_flat = pd.concat([df["res1_three"], df["res2_three"]], axis=0).dropna()
		aa_flat = aa_flat[aa_flat != ""]
		aa_dist = aa_flat.value_counts().rename_axis("res_three").reset_index(name="count")
		aa_dist.to_csv(os.path.join(dirs["csv"], "amino_acid_counts.csv"), index=False)

	# Overall amino-acid distribution across all residues (not just endpoints)
	if len(aa_overall_counts) > 0:
		df_aa_all = pd.DataFrame({"res_three": list(aa_overall_counts.keys()), "count": list(aa_overall_counts.values())})
		df_aa_all = df_aa_all.sort_values("count", ascending=False)
		df_aa_all.to_csv(os.path.join(dirs["csv"], "amino_acid_overall_counts.csv"), index=False)

		# Physicochemical properties distribution
		df["res1_prop"] = df["res1_one"].map(lambda x: label_residue_property(x))
		df["res2_prop"] = df["res2_one"].map(lambda x: label_residue_property(x))
		prop_counts = pd.concat([
			df["res1_prop"].value_counts(),
			df["res2_prop"].value_counts(),
		], axis=1, keys=["p1", "p2"]).fillna(0.0).sum(axis=1).rename_axis("property").reset_index(name="count")
		prop_counts.to_csv(os.path.join(dirs["csv"], "residue_property_counts.csv"), index=False)

		# Geometry distributions (include chem type and group-specific angles/dihedrals)
		df_geom = df[["link_type", "chem_type", "bond_length", "bond_angle", "dihedral", "angle_i", "angle_j", "dihedral_1", "dihedral_2"]].copy()
		df_geom.to_csv(os.path.join(dirs["csv"], "geometry_by_linktype.csv"), index=False)

		# Secondary structure distribution
		if not args.skip_secondary:
			ss_flat = pd.concat([
				df["ss1"].value_counts(),
				df["ss2"].value_counts(),
			], axis=1, keys=["p1", "p2"]).fillna(0.0).sum(axis=1).rename_axis("ss").reset_index(name="count")
			ss_flat.to_csv(os.path.join(dirs["csv"], "secondary_structure_counts.csv"), index=False)

		# SASA distributions (buried vs exposed)
		exposure_flat = pd.concat([
			df["exposure1"].value_counts(),
			df["exposure2"].value_counts(),
		], axis=1, keys=["p1", "p2"]).fillna(0.0).sum(axis=1).rename_axis("exposure").reset_index(name="count")
		exposure_flat.to_csv(os.path.join(dirs["csv"], "exposure_counts.csv"), index=False)

		# Sequence span distribution (same chain only)
		df_span = df[df["same_chain"] & df["sequence_span"].notna()].copy()
		# Overall same-chain span distribution
		df_span[["sequence_span"]].to_csv(os.path.join(dirs["csv"], "sequence_span.csv"), index=False)
		# Same-chain span distribution by chemical type (e.g. disulfide / lactone / amide / other)
		if "chem_type" in df_span.columns:
			df_span[["sequence_span", "chem_type"]].to_csv(
				os.path.join(dirs["csv"], "sequence_span_by_chem.csv"),
				index=False,
			)

	# Plots: reuse CSV-based helper so normal模式和 --csv_dir 模式完全一致
	try:
		replot_from_csv(dirs["csv"], dirs["plots"])
	except Exception:
		pass

	print(f"Done. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
	main()


