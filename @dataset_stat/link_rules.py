import os
from typing import Dict, Optional, Tuple, List
import csv


Rule = Dict[str, Optional[str]]


class LinkRuleBook:
	"""
	Load link.csv and provide:
	- rule lookup by (res1,res2,atom1,atom2) with symmetric fallback
	- anchors for angles/dihedrals
	- heuristic chemical type classification (disulfide / lactone / amide / other)
	"""
	def __init__(self, csv_path: str):
		self.csv_path = csv_path
		self.rules: List[Rule] = []
		self._load()

	def _load(self) -> None:
		if not self.csv_path or not os.path.exists(self.csv_path):
			return
		with open(self.csv_path, "r", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			for row in reader:
				# Normalize to uppercase 3-letter AA and atom names
				r: Rule = {
					"res1": (row.get("res1") or "").strip().upper(),
					"res2": (row.get("res2") or "").strip().upper(),
					"atom1": (row.get("atom1") or "").strip().upper(),
					"atom2": (row.get("atom2") or "").strip().upper(),
					"angle_i_anchor": (row.get("angle_i_anchor") or "").strip().upper() or None,
					"angle_j_anchor": (row.get("angle_j_anchor") or "").strip().upper() or None,
					"dihedral_1_anchor_i": (row.get("dihedral_1_anchor_i") or "").strip().upper() or None,
					"dihedral_1_anchor_j": (row.get("dihedral_1_anchor_j") or "").strip().upper() or None,
					"dihedral_2_anchor_i": (row.get("dihedral_2_anchor_i") or "").strip().upper() or None,
					"dihedral_2_anchor_j": (row.get("dihedral_2_anchor_j") or "").strip().upper() or None,
				}
				self.rules.append(r)

	def _match(self, res1: str, res2: str, atom1: str, atom2: str) -> Optional[Rule]:
		# exact
		for r in self.rules:
			if r["res1"] == res1 and r["res2"] == res2 and r["atom1"] == atom1 and r["atom2"] == atom2:
				return r
		# allow ALL wildcard
		for r in self.rules:
			if (r["res1"] in (res1, "ALL")) and (r["res2"] in (res2, "ALL")) and r["atom1"] == atom1 and r["atom2"] == atom2:
				return r
		return None

	def find_rule(self, res1_three: str, res2_three: str, atom1: str, atom2: str) -> Tuple[Optional[Rule], bool]:
		"""
		Return (rule, swapped). If not found, try swapped orientation.
		"""
		res1 = (res1_three or "").upper()
		res2 = (res2_three or "").upper()
		a1 = (atom1 or "").upper()
		a2 = (atom2 or "").upper()
		r = self._match(res1, res2, a1, a2)
		if r:
			return r, False
		r2 = self._match(res2, res1, a2, a1)
		if r2:
			return r2, True
		return None, False

	@staticmethod
	def classify_chem_type(res1: str, res2: str, atom1: str, atom2: str, conn_type_id: str = "") -> str:
		"""
		Heuristic chemical type:
		- disulfide: CYS SG – CYS SG
		- lactone: (ASP/GLU) CG/CD – (SER/THR/TYR) OG/OG1/OH (either order)
		- amide: N/NZ – CG/CD of (ASP/GLU/ASN/GLN) or generic C–N (non-canonical peptide excluded elsewhere)
		Otherwise try conn_type_id hints.
		"""
		r1 = (res1 or "").upper()
		r2 = (res2 or "").upper()
		a1 = (atom1 or "").upper()
		a2 = (atom2 or "").upper()
		t = (conn_type_id or "").lower()

		if r1 == "CYS" and r2 == "CYS" and a1 == "SG" and a2 == "SG":
			return "disulfide"

		def is_acid_c(res: str, atom: str) -> bool:
			return (res in {"ASP", "GLU"} and atom in {"CG", "CD"})

		def is_alcohol_o(res: str, atom: str) -> bool:
			return (res in {"SER", "THR", "TYR"} and atom in {"OG", "OG1", "OH"})

		if (is_acid_c(r1, a1) and is_alcohol_o(r2, a2)) or (is_acid_c(r2, a2) and is_alcohol_o(r1, a1)):
			return "lactone"

		def is_amide_like_n(res: str, atom: str) -> bool:
			return atom in {"N", "NZ"}

		def is_carbonyl_c(res: str, atom: str) -> bool:
			return (res in {"ASP", "GLU", "ASN", "GLN"} and atom in {"CG", "CD"}) or atom == "C"

		if (is_amide_like_n(r1, a1) and is_carbonyl_c(r2, a2)) or (is_amide_like_n(r2, a2) and is_carbonyl_c(r1, a1)):
			return "amide"

		if "disulf" in t:
			return "disulfide"
		if "amide" in t or "isopep" in t:
			return "amide"

		return "other"


