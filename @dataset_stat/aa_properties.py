AA_THREE_TO_ONE = {
	"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
	"GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
	"LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
	"SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Only standard 20 amino acids
STANDARD_AA_THREE = set(AA_THREE_TO_ONE.keys())
STANDARD_AA_ONE = set(AA_THREE_TO_ONE[a] for a in STANDARD_AA_THREE)

AA_CLASS = {
	# Hydrophobic (nonpolar)
	"A": {"hydrophobic", "nonpolar"},
	"V": {"hydrophobic", "nonpolar"},
	"I": {"hydrophobic", "nonpolar"},
	"L": {"hydrophobic", "nonpolar"},
	"M": {"hydrophobic", "nonpolar"},
	"F": {"hydrophobic", "nonpolar", "aromatic"},
	"W": {"hydrophobic", "nonpolar", "aromatic"},
	"P": {"hydrophobic", "nonpolar"},
	# Polar uncharged
	"S": {"polar", "uncharged"},
	"T": {"polar", "uncharged"},
	"N": {"polar", "uncharged"},
	"Q": {"polar", "uncharged"},
	"Y": {"polar", "uncharged", "aromatic"},
	"C": {"polar", "uncharged"},
	"G": {"polar", "uncharged"},
	# Charged
	"D": {"polar", "negative", "charged"},
	"E": {"polar", "negative", "charged"},
	"K": {"polar", "positive", "charged"},
	"R": {"polar", "positive", "charged"},
	"H": {"polar", "positive", "charged"},
}

# Approximate maximum ASA (Å^2) for relative SASA normalization (values from Tien et al., 2013)
MAX_ASA = {
	"A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
	"Q": 214.0, "E": 214.0, "G": 97.0,  "H": 216.0, "I": 195.0,
	"L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
	"S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}

def aa_one(three: str) -> str:
	three = (three or "").upper()
	return AA_THREE_TO_ONE.get(three, "X")

def is_standard_three(three: str) -> bool:
	return (three or "") in STANDARD_AA_THREE

def residue_classes(one: str) -> set:
	return AA_CLASS.get(one, set())



