from typing import Dict, Tuple

from aa_properties import residue_classes


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def classify_endpoint(
	chain_id: str,
	label_seq_id: int,
	atom_name: str,
	chain_boundaries: Dict[str, Tuple[int, int]],
) -> str:
	"""
	Classify an endpoint into head/tail/sidechain using chain boundaries and atom identity.
	- head: residue is chain N-terminus AND atom is N
	- tail: residue is chain C-terminus AND atom is C or OXT
	- sidechain: otherwise
	"""
	lo, hi = chain_boundaries.get(chain_id, (None, None))
	aname = (atom_name or "").upper()
	if lo is not None and label_seq_id == lo and aname == "N":
		return "head"
	if hi is not None and label_seq_id == hi and aname in {"C", "OXT"}:
		return "tail"
	return "sidechain"


def classify_pair(t1: str, t2: str) -> str:
	parts = sorted([t1, t2], key=lambda x: {"head": 0, "tail": 1, "sidechain": 2}.get(x, 3))
	return f"{parts[0]}-{parts[1]}"


def label_residue_property(one: str) -> str:
	classes = residue_classes((one or "X").upper())
	if "hydrophobic" in classes:
		return "hydrophobic"
	if "positive" in classes or "negative" in classes or "charged" in classes:
		return "charged"
	if "aromatic" in classes:
		return "aromatic"
	if "polar" in classes:
		return "polar"
	return "other"


