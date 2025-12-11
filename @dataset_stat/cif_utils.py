from typing import Dict, List, Tuple, Any, Optional

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import MMCIFParser
from Bio.PDB.DSSP import DSSP

from aa_properties import AA_THREE_TO_ONE, is_standard_three, aa_one


def load_mmcif_dict(path: str) -> Dict[str, Any]:
	return MMCIF2Dict(path)


def _as_list(d: Dict[str, Any], key: str) -> List[Any]:
	val = d.get(key, [])
	if isinstance(val, list):
		return val
	return [val]


def build_atom_index(mmcif: Dict[str, Any]) -> Tuple[
	Dict[Tuple[str, int, str], Tuple[float, float, float]],
	Dict[Tuple[str, int], str],
	Dict[str, str],
]:
	"""
	Returns:
	- atom_index: (label_asym_id, label_seq_id, label_atom_id) -> (x, y, z)
	- residue_name: (label_asym_id, label_seq_id) -> 3-letter residue name
	- chain_sequences: chain_id -> one-letter sequence (label scheme, ordered by label_seq_id)
	"""
	label_asym_id = _as_list(mmcif, "_atom_site.label_asym_id")
	label_seq_id = _as_list(mmcif, "_atom_site.label_seq_id")
	label_atom_id = _as_list(mmcif, "_atom_site.label_atom_id")
	label_comp_id = _as_list(mmcif, "_atom_site.label_comp_id")
	auth_seq_id = _as_list(mmcif, "_atom_site.auth_seq_id")
	x = _as_list(mmcif, "_atom_site.Cartn_x")
	y = _as_list(mmcif, "_atom_site.Cartn_y")
	z = _as_list(mmcif, "_atom_site.Cartn_z")
	alt_id = _as_list(mmcif, "_atom_site.label_alt_id") if "_atom_site.label_alt_id" in mmcif else ["." for _ in label_asym_id]

	atom_index: Dict[Tuple[str, int, str], Tuple[float, float, float]] = {}
	residue_name: Dict[Tuple[str, int], str] = {}
	chain_positions: Dict[str, List[Tuple[int, str]]] = {}
	for i in range(len(label_asym_id)):
		if alt_id[i] not in (".", "A", "?"):
			continue
		chain = str(label_asym_id[i])
		try:
			seq = int(label_seq_id[i])
		except Exception:
			continue
		atom = str(label_atom_id[i]).upper()
		comp = str(label_comp_id[i]).upper()
		if not is_standard_three(comp):
			continue
		try:
			coord = (float(x[i]), float(y[i]), float(z[i]))
		except Exception:
			continue
		atom_index[(chain, seq, atom)] = coord
		residue_name[(chain, seq)] = comp
		one = aa_one(comp)
		chain_positions.setdefault(chain, []).append((seq, one))
	# Build sequences per chain in order of label_seq_id
	chain_sequences: Dict[str, str] = {}
	for ch, lst in chain_positions.items():
		lst_sorted = sorted(set(lst), key=lambda t: t[0])
		chain_sequences[ch] = "".join([one for (_, one) in lst_sorted])
	return atom_index, residue_name, chain_sequences


def get_chain_boundaries(mmcif: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
	label_asym_id = _as_list(mmcif, "_atom_site.label_asym_id")
	label_seq_id = _as_list(mmcif, "_atom_site.label_seq_id")
	label_comp_id = _as_list(mmcif, "_atom_site.label_comp_id")
	alt_id = _as_list(mmcif, "_atom_site.label_alt_id") if "_atom_site.label_alt_id" in mmcif else ["." for _ in label_asym_id]
	positions: Dict[str, List[int]] = {}
	for i in range(len(label_asym_id)):
		if alt_id[i] not in (".", "A", "?"):
			continue
		comp = str(label_comp_id[i]).upper()
		if not is_standard_three(comp):
			continue
		chain = str(label_asym_id[i])
		try:
			seq = int(label_seq_id[i])
		except Exception:
			continue
		positions.setdefault(chain, []).append(seq)
	bounds: Dict[str, Tuple[int, int]] = {}
	for ch, vals in positions.items():
		if not vals:
			continue
		bounds[ch] = (min(vals), max(vals))
	return bounds


def parse_struct_conns(mmcif: Dict[str, Any]) -> List[Dict[str, Any]]:
	"""
	Parse _struct_conn, retaining covalent-like types; exclude metal/ionic/hydrogen.
	Returned rows contain label_* fields when available; fallback to auth_* if label is missing.
	"""
	if "_struct_conn.id" not in mmcif:
		return []
	n = len(_as_list(mmcif, "_struct_conn.id"))
	def get_col(prefix: str) -> List[Any]:
		key = f"_struct_conn.{prefix}"
		return _as_list(mmcif, key) if key in mmcif else ["?" for _ in range(n)]
	def partner_cols(ptnr: str, suffix: str) -> List[Any]:
		key = f"_struct_conn.p{ptnr}_{suffix}"
		return _as_list(mmcif, key) if key in mmcif else ["?" for _ in range(n)]
	conn_id = get_col("id")
	conn_type = get_col("conn_type_id")
	pt1_asym = partner_cols("tnr1", "label_asym_id")
	pt1_seq = partner_cols("tnr1", "label_seq_id")
	pt1_atom = partner_cols("tnr1", "label_atom_id")
	pt1_comp = partner_cols("tnr1", "label_comp_id")
	pt2_asym = partner_cols("tnr2", "label_asym_id")
	pt2_seq = partner_cols("tnr2", "label_seq_id")
	pt2_atom = partner_cols("tnr2", "label_atom_id")
	pt2_comp = partner_cols("tnr2", "label_comp_id")

	rows: List[Dict[str, Any]] = []
	for i in range(n):
		t = str(conn_type[i] or "").lower()
		if any(x in t for x in ["metal", "ionic", "hydrog"]):
			continue
		if not (("coval" in t) or ("disulf" in t) or ("amide" in t) or ("thio" in t) or ("cross" in t) or ("isopep" in t)):
			continue
		try:
			seq1 = int(pt1_seq[i])
			seq2 = int(pt2_seq[i])
		except Exception:
			continue
		rows.append({
			"id": conn_id[i],
			"type": conn_type[i],
			"p1": {
				"chain": str(pt1_asym[i]),
				"seq": seq1,
				"atom": str(pt1_atom[i]).upper(),
				"comp": str(pt1_comp[i]).upper(),
			},
			"p2": {
				"chain": str(pt2_asym[i]),
				"seq": seq2,
				"atom": str(pt2_atom[i]).upper(),
				"comp": str(pt2_comp[i]).upper(),
			},
		})
	return rows


def parse_secondary_structure_sets(mmcif: Dict[str, Any]) -> Tuple[set, set]:
	"""
	Return sets of (chain, label_seq_id) in helix and sheet.
	"""
	helix = set()
	sheet = set()
	if "_struct_conf.conf_type_id" in mmcif:
		n = len(_as_list(mmcif, "_struct_conf.conf_type_id"))
		conf_type = _as_list(mmcif, "_struct_conf.conf_type_id")
		beg_asym = _as_list(mmcif, "_struct_conf.beg_label_asym_id")
		beg_seq = _as_list(mmcif, "_struct_conf.beg_label_seq_id")
		end_seq = _as_list(mmcif, "_struct_conf.end_label_seq_id")
		for i in range(n):
			t = str(conf_type[i] or "").lower()
			if "helix" in t:
				chain = str(beg_asym[i])
				try:
					bs = int(beg_seq[i])
					es = int(end_seq[i])
				except Exception:
					continue
				for s in range(min(bs, es), max(bs, es) + 1):
					helix.add((chain, s))
	if "_struct_sheet_range.id" in mmcif:
		n = len(_as_list(mmcif, "_struct_sheet_range.id"))
		asym = _as_list(mmcif, "_struct_sheet_range.beg_label_asym_id")
		beg = _as_list(mmcif, "_struct_sheet_range.beg_label_seq_id")
		end = _as_list(mmcif, "_struct_sheet_range.end_label_seq_id")
		for i in range(n):
			chain = str(asym[i])
			try:
				bs = int(beg[i])
				es = int(end[i])
			except Exception:
				continue
			for s in range(min(bs, es), max(bs, es) + 1):
				sheet.add((chain, s))
	return helix, sheet


def compute_dssp_secondary_structure_sets(cif_path: str) -> Tuple[set, set]:
	"""
	Compute helix/sheet sets using DSSP as a fallback when CIF lacks annotations.
	Returns sets of (chain_id, seq_id) matching Bio.PDB residue numbering used by MMCIFParser.
	Note: Requires 'mkdssp' (conda package 'dssp') available in PATH.
	"""
	helix = set()
	sheet = set()
	try:
		parser = MMCIFParser(QUIET=True)
		struct = parser.get_structure("struct", cif_path)
		model = next(struct.get_models())
	except Exception:
		return helix, sheet
	try:
		# Explicitly set file type for mmCIF to improve residue handling
		dssp = DSSP(model, cif_path, file_type="MMCIF")
	except Exception:
		# DSSP unavailable or failed
		return helix, sheet
	try:
		helix_codes = {"H", "G", "I"}
		sheet_codes = {"E", "B"}
		for key in dssp.keys():
			try:
				chain_id, res_id = key
				ss = dssp[key][2]  # Secondary structure letter
				# res_id is a tuple like (' ', seq, icode)
				seq = int(res_id[1]) if isinstance(res_id, tuple) else int(res_id)
				if ss in helix_codes:
					helix.add((str(chain_id), seq))
				elif ss in sheet_codes:
					sheet.add((str(chain_id), seq))
			except Exception:
				continue
	except Exception:
		# Be robust against API differences
		return helix, sheet
	return helix, sheet


