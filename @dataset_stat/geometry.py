import math
from typing import Optional, Sequence

import numpy as np


def vector(p1: Sequence[float], p2: Sequence[float]) -> np.ndarray:
	return np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)


def norm(v: Sequence[float]) -> float:
	return float(np.linalg.norm(v))


def unit(v: Sequence[float]) -> np.ndarray:
	v = np.asarray(v, dtype=float)
	n = np.linalg.norm(v)
	if n == 0.0:
		return v
	return v / n


def distance(p1: Sequence[float], p2: Sequence[float]) -> float:
	return norm(vector(p1, p2))


def angle(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> float:
	"""
	Return angle in degrees for angle p1-p2-p3
	"""
	v1 = unit(vector(p2, p1))
	v2 = unit(vector(p2, p3))
	cosang = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
	return math.degrees(math.acos(cosang))


def dihedral(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float], p4: Sequence[float]) -> float:
	"""
	Return dihedral angle in degrees for p1-p2-p3-p4
	Based on vector and cross product formulation.
	"""
	p1 = np.asarray(p1, dtype=float)
	p2 = np.asarray(p2, dtype=float)
	p3 = np.asarray(p3, dtype=float)
	p4 = np.asarray(p4, dtype=float)
	b0 = -(p2 - p1)
	b1 = p3 - p2
	b2 = p4 - p3
	# Normalize b1 so that it does not influence magnitude of vector
	b1_u = unit(b1)
	# Compute normals
	v = b0 - np.dot(b0, b1_u) * b1_u
	w = b2 - np.dot(b2, b1_u) * b1_u
	x = float(np.dot(unit(v), unit(w)))
	y = float(np.dot(np.cross(b1_u, unit(v)), unit(w)))
	angle_rad = math.atan2(y, x)
	return math.degrees(angle_rad)



