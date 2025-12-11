from typing import Optional
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def barplot_counts(df: pd.DataFrame, x: str, count_name: str, 
				title: str, out_png: str, log_y: bool = False, rotation: Optional[float] = 45) -> None:
	ensure_dir(os.path.dirname(out_png))
	plt.figure(figsize=(8, 5))
	ax = sns.barplot(data=df, x=x, y=count_name)
	if log_y:
		try:
			ax.set_yscale("log")
		except Exception:
			pass
	# Title / labels / ticks / legend font sizes
	ax.set_title(title, fontsize=16)
	ax.set_xlabel(ax.get_xlabel(), fontsize=14)
	ax.set_ylabel(ax.get_ylabel(), fontsize=14)
	for tick in ax.get_xticklabels():
		tick.set_rotation(rotation)
	ax.tick_params(axis="both", labelsize=14)
	leg = ax.get_legend()
	if leg is not None:
		if leg.get_title() is not None:
			leg.set_title(leg.get_title().get_text(), prop={"size": 12})
		for txt in leg.get_texts():
			txt.set_fontsize(12)
	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close()


def histplot(df: pd.DataFrame, col: str, title: str, out_png: str, bins: int = 50, log: bool = False, xlim: Optional[tuple] = None) -> None:
	ensure_dir(os.path.dirname(out_png))
	plt.figure(figsize=(7, 5))
	ax = sns.histplot(df[col].dropna(), bins=bins, kde=False, log_scale=log)
	ax.set_title(title, fontsize=16)
	ax.set_xlabel(ax.get_xlabel(), fontsize=14)
	ax.set_ylabel(ax.get_ylabel(), fontsize=14)
	ax.tick_params(axis="both", labelsize=14)
	if xlim:
		ax.set_xlim(*xlim)
	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close()

def overlay_kde_by_hue_maxnorm(
	df: pd.DataFrame,
	value_col: str,
	hue_col: str,
	title: str,
	out_png: str,
	xlim: Optional[tuple] = None,
	bw_adjust: float = 0.8,
	palette: Optional[dict] = None,
	clip_quantiles: Optional[tuple] = (0.01, 0.99),
) -> None:
	"""
	Draw KDE lines per group (hue) and normalize each group's curve to max=1 (0-1 y-scale).
	"""
	ensure_dir(os.path.dirname(out_png))
	plt.figure(figsize=(8, 5))
	data = df.dropna(subset=[value_col, hue_col]).copy()
	if data.empty:
		return
	if clip_quantiles is not None:
		try:
			ql, qh = clip_quantiles
			low = data[value_col].quantile(ql)
			high = data[value_col].quantile(qh)
			data = data[(data[value_col] >= low) & (data[value_col] <= high)]
		except Exception:
			pass
	hues = [h for h in data[hue_col].unique().tolist() if pd.notna(h)]
	ax = plt.gca()
	for h in hues:
		sub = data[data[hue_col] == h]
		if sub.empty:
			continue
		try:
			sns.kdeplot(
				data=sub,
				x=value_col,
				bw_adjust= bw_adjust,
				fill=False,
				alpha=1.0,
				color=(palette or {}).get(h, None),
				label=str(h),
				ax=ax,
			)
			# Normalize the last added line to max=1
			if ax.lines:
				line = ax.lines[-1]
				ydata = line.get_ydata()
				if ydata is not None and len(ydata) > 0:
					mx = float(np.max(ydata))
					if mx > 0:
						line.set_ydata(ydata / mx)
		except Exception:
			continue
	# Title / labels / ticks / legend font sizes
	ax.set_title(title, fontsize=16)
	ax.set_ylabel("Normalized density (max=1)", fontsize=14)
	ax.set_xlabel(ax.get_xlabel(), fontsize=14)
	ax.tick_params(axis="both", labelsize=14)
	if xlim:
		ax.set_xlim(*xlim)
	ax.set_ylim(0.0, 1.0)
	leg = ax.get_legend()
	if leg is None:
		leg = ax.legend()
	if leg is not None:
		if leg.get_title() is not None:
			leg.set_title(leg.get_title().get_text(), prop={"size": 12})
		for txt in leg.get_texts():
			txt.set_fontsize(12)
	plt.tight_layout()
	plt.savefig(out_png, dpi=220)
	plt.close()

def countplot(df: pd.DataFrame, col: str, title: str, out_png: str, order: Optional[list] = None) -> None:
	ensure_dir(os.path.dirname(out_png))
	plt.figure(figsize=(7, 5))
	ax = sns.countplot(data=df, x=col, order=order)
	for tick in ax.get_xticklabels():
		tick.set_rotation(45)
	ax.set_title(title, fontsize=16)
	ax.set_xlabel(ax.get_xlabel(), fontsize=14)
	ax.set_ylabel(ax.get_ylabel(), fontsize=14)
	ax.tick_params(axis="both", labelsize=14)
	leg = ax.get_legend()
	if leg is not None:
		if leg.get_title() is not None:
			leg.set_title(leg.get_title().get_text(), prop={"size": 12})
		for txt in leg.get_texts():
			txt.set_fontsize(12)
	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close()


def overlay_hist_by_hue(
	df: pd.DataFrame,
	value_col: str,
	hue_col: str,
	title: str,
	out_png: str,
	bins: int = 60,
	xlim: Optional[tuple] = None,
	palette: Optional[dict] = None,
	stat: str = "count",
	clip_quantiles: Optional[tuple] = None,
) -> None:
	ensure_dir(os.path.dirname(out_png))
	plt.figure(figsize=(8, 5))
	data = df.dropna(subset=[value_col]).copy()
	if clip_quantiles is not None:
		try:
			ql, qh = clip_quantiles
			low = data[value_col].quantile(ql)
			high = data[value_col].quantile(qh)
			data = data[(data[value_col] >= low) & (data[value_col] <= high)]
		except Exception:
			pass
	ax = sns.histplot(
		data=data,
		x=value_col,
		hue=hue_col,
		bins=bins,
		element="step",
		stat=stat,
		multiple="layer",
		palette=palette,
	)
	ax.set_title(title, fontsize=16)
	ax.set_xlabel(ax.get_xlabel(), fontsize=14)
	ax.set_ylabel(ax.get_ylabel(), fontsize=14)
	ax.tick_params(axis="both", labelsize=14)
	if xlim:
		ax.set_xlim(*xlim)
	leg = ax.get_legend()
	if leg is None:
		leg = ax.legend()
	if leg is not None:
		if leg.get_title() is not None:
			leg.set_title(leg.get_title().get_text(), prop={"size": 12})
		for txt in leg.get_texts():
			txt.set_fontsize(12)
	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close()


def pieplot_from_counts(df_counts: pd.DataFrame, label_col: str, count_col: str, title: str, out_png: str) -> None:
	ensure_dir(os.path.dirname(out_png))
	plt.figure(figsize=(6, 6))
	labels = df_counts[label_col].tolist()
	sizes = df_counts[count_col].astype(float).tolist()
	# 控制饼图标签和数值文字大小
	plt.pie(
		sizes,
		labels=labels,
		autopct="%1.1f%%",
		startangle=140,
		counterclock=False,
		textprops={"fontsize": 14},
	)
	plt.title(title, fontsize=16)
	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close()


def overlay_kde_by_hue(
	df: pd.DataFrame,
	value_col: str,
	hue_col: str,
	title: str,
	out_png: str,
	xlim: Optional[tuple] = None,
	bw_adjust: float = 0.8,
	palette: Optional[dict] = None,
	clip_quantiles: Optional[tuple] = (0.01, 0.99),
	fill_alpha: float = 0.35,
	ax_label: Optional[str] = None,
) -> None:
	"""
	Smoother density curves per group (hue). Useful for angle/dihedral/length distributions.
	"""
	ensure_dir(os.path.dirname(out_png))
	plt.figure(figsize=(8, 5))
	data = df.dropna(subset=[value_col]).copy()
	if clip_quantiles is not None:
		try:
			ql, qh = clip_quantiles
			low = data[value_col].quantile(ql)
			high = data[value_col].quantile(qh)
			data = data[(data[value_col] >= low) & (data[value_col] <= high)]
		except Exception:
			pass
	ax = sns.kdeplot(
		data=data,
		x=value_col,
		hue=hue_col,
		common_norm=False,
		bw_adjust=bw_adjust,
		fill=True,
		alpha=fill_alpha,
		palette=palette,
	)
	# Title / labels / ticks / legend font sizes
	ax.set_title(title, fontsize=16)
	ax.set_ylabel("Density", fontsize=14)
	if ax_label is not None:
		ax.set_xlabel(ax_label, fontsize=14)
	else:
		ax.set_xlabel(ax.get_xlabel(), fontsize=14)
	ax.tick_params(axis="both", labelsize=14)
	# Ensure legend is drawn and styled
	leg = ax.get_legend()
	if leg is None:
		leg = ax.legend()
	if leg is not None:
		if leg.get_title() is not None:
			leg.set_title(leg.get_title().get_text(), prop={"size": 12})
		for txt in leg.get_texts():
			txt.set_fontsize(12)
	if xlim:
		ax.set_xlim(*xlim)
	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close()


def combined_kde_groups_by_chem(
	df: pd.DataFrame,
	value_cols: list,
	group_labels: list,
	chem_col: str,
	title: str,
	out_png: str,
	xlim: Optional[tuple] = None,
	bw_adjust: float = 0.8,
	label_map: Optional[dict] = None,
	skip_second_for_disulfide: bool = True,
	palette: Optional[dict] = None,
	clip_quantiles: Optional[tuple] = (0.01, 0.99),
	fill_alpha: float = 0.35,
	ax_label = None,
) -> None:
	"""
	Draw KDE lines for multiple value columns (groups) overlaid in a single figure,
	colored by chemical type and styled by group label.
	- value_cols: e.g., ["angle_i", "angle_j"] or ["dihedral_1", "dihedral_2"]
	- group_labels: e.g., ["i-anchor", "j-anchor"] or ["group1", "group2"]
	Legend entries will be like "amide - i-anchor".
	"""
	assert len(value_cols) == len(group_labels)
	ensure_dir(os.path.dirname(out_png))
	plt.figure(figsize=(9, 6))
	linestyles = ["solid", "dashed", "dashdot", "dotted"]
	chem_types = ["disulfide", "lactone", "amide"]
	default_palette = {"disulfide": "#FF7F0E", "lactone": "#D62728", "amide": "#1F77B4"}
	colors = palette or default_palette
	# Pre-compute clip bounds per value column (global across chems)
	bounds: dict = {}
	if clip_quantiles is not None:
		for col in value_cols:
			try:
				col_data = df[col].dropna()
				if not col_data.empty:
					ql, qh = clip_quantiles
					low = col_data.quantile(ql)
					high = col_data.quantile(qh)
					bounds[col] = (low, high)
			except Exception:
				continue
	for gi, (col, glabel) in enumerate(zip(value_cols, group_labels)):
		style = linestyles[gi % len(linestyles)]
		for chem in chem_types:
			if skip_second_for_disulfide and chem == "disulfide" and gi > 0:
				continue
			sub = df[(df[chem_col] == chem) & df[col].notna()].copy()
			if clip_quantiles is not None and col in bounds:
				low, high = bounds[col]
				sub = sub[(sub[col] >= low) & (sub[col] <= high)]
			if sub.empty:
				continue
			lbl = f"{chem} - {glabel}"
			if label_map and chem in label_map and glabel in label_map[chem]:
				lbl = f"{chem} - {label_map[chem][glabel]}"
			try:
				sns.kdeplot(
					data=sub,
					x=col,
					common_norm=False,
					bw_adjust=bw_adjust,
					fill=True,
					alpha=fill_alpha,
					color=colors.get(chem, None),
					label=lbl,
					linestyle=style,
				)
			except Exception:
				continue
	ax = plt.gca()
	ax.set_title(title, fontsize=16)
	ax.set_ylabel("Density", fontsize=14)
	if ax_label is not None:
		ax.set_xlabel(ax_label, fontsize=14)
	else:
		ax.set_xlabel(ax.get_xlabel(), fontsize=14)
	ax.tick_params(axis="both", labelsize=14)
	if xlim:
		ax.set_xlim(*xlim)
	# Ensure legend is drawn and styled
	leg = ax.get_legend()
	if leg is None:
		leg = ax.legend()
	if leg is not None:
		if leg.get_title() is not None:
			leg.set_title(leg.get_title().get_text(), prop={"size": 12})
		for txt in leg.get_texts():
			txt.set_fontsize(12)
	plt.tight_layout()
	plt.savefig(out_png, dpi=220)
	plt.close()


def kdeplot_simple(
	series: pd.Series,
	title: str,
	out_png: str,
	xlabel: str = "",
	xlim: Optional[tuple] = None,
	bw_adjust: float = 0.8,
	clip_quantiles: Optional[tuple] = (0.01, 0.99),
	fill_alpha: float = 0.35,
) -> None:
	ensure_dir(os.path.dirname(out_png))
	s = series.dropna().astype(float)
	if s.empty:
		return
	if clip_quantiles is not None:
		try:
			ql, qh = clip_quantiles
			low = s.quantile(ql)
			high = s.quantile(qh)
			s = s[(s >= low) & (s <= high)]
		except Exception:
			pass
	plt.figure(figsize=(8, 5))
	try:
		ax = sns.kdeplot(x=s, bw_adjust=bw_adjust, fill=True, alpha=fill_alpha)
	except Exception:
		return
	ax.set_title(title, fontsize=16)
	ax.set_ylabel("Density", fontsize=14)
	if xlabel:
		ax.set_xlabel(xlabel, fontsize=14)
	else:
		ax.set_xlabel(ax.get_xlabel(), fontsize=14)
	ax.tick_params(axis="both", labelsize=14)
	if xlim:
		ax.set_xlim(*xlim)
	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close()


def hist_density_simple(
	series: pd.Series,
	title: str,
	out_png: str,
	xlabel: str = "",
	xlim: Optional[tuple] = None,
	bins: Optional[object] = "auto",
	clip_quantiles: Optional[tuple] = (0.01, 0.99),
	alpha: float = 0.45,
	color: str = "#1F77B4",
	y_label: str = "Density",
) -> None:
	ensure_dir(os.path.dirname(out_png))
	s = series.dropna().astype(float)
	if s.empty:
		return
	# Remove negatives for spans
	s = s[s >= 0.0]
	if clip_quantiles is not None:
		try:
			ql, qh = clip_quantiles
			low = s.quantile(ql)
			high = s.quantile(qh)
			s = s[(s >= low) & (s <= high)]
		except Exception:
			pass
	if s.empty:
		return
	plt.figure(figsize=(8, 5))
	try:
		ax = sns.histplot(x=s, bins=bins, stat="density", color=color, alpha=alpha)
	except Exception:
		return
	ax.set_title(title, fontsize=16)
	if y_label:
		ax.set_ylabel(y_label, fontsize=14)
	if xlabel:
		ax.set_xlabel(xlabel, fontsize=14)
	else:
		ax.set_xlabel(ax.get_xlabel(), fontsize=14)
	ax.tick_params(axis="both", labelsize=14)
	if xlim:
		ax.set_xlim(*xlim)
	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close()


def hist_density_with_ybreak(
	series: pd.Series,
	title: str,
	out_png: str,
	xlabel: str = "",
	xlim: Optional[tuple] = None,
	bins: Optional[object] = "auto",
	clip_quantiles: Optional[tuple] = (0.01, 0.99),
	alpha: float = 0.45,
	color: str = "#1F77B4",
	y_break: Optional[tuple] = None,
	y_upper: Optional[float] = None,
) -> None:
	"""
	直方图 + 密度（同 hist_density_simple），支持在 y 轴上做一个断轴。
	- y_break: (low, high)，表示 [low, high] 这段被“跳过”（中间用斜线标记）。
	"""
	# 如果没有设置断轴，直接退回到原始实现
	if not y_break:
		hist_density_simple(
			series=series,
			title=title,
			out_png=out_png,
			xlabel=xlabel,
			xlim=xlim,
			bins=bins,
			clip_quantiles=clip_quantiles,
			alpha=alpha,
			color=color,
		)
		return

	ensure_dir(os.path.dirname(out_png))
	s = series.dropna().astype(float)
	if s.empty:
		return
	# Remove negatives for spans
	s = s[s >= 0.0]
	if clip_quantiles is not None:
		try:
			ql, qh = clip_quantiles
			low_q = s.quantile(ql)
			high_q = s.quantile(qh)
			s = s[(s >= low_q) & (s <= high_q)]
		except Exception:
			pass
	if s.empty:
		return

	# 解析 y 轴断轴范围
	try:
		y_low, y_high = float(y_break[0]), float(y_break[1])
	except Exception:
		# 解析失败就退回正常画法
		hist_density_simple(
			series=series,
			title=title,
			out_png=out_png,
			xlabel=xlabel,
			xlim=xlim,
			bins=bins,
			clip_quantiles=clip_quantiles,
			alpha=alpha,
			color=color,
		)
		return
	if not (0.0 < y_low < y_high):
		hist_density_simple(
			series=series,
			title=title,
			out_png=out_png,
			xlabel=xlabel,
			xlim=xlim,
			bins=bins,
			clip_quantiles=clip_quantiles,
			alpha=alpha,
			color=color,
		)
		return

	fig, (ax_top, ax_bottom) = plt.subplots(
		2,
		1,
		sharex=True,
		figsize=(8, 5),
		gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2},
	)
	try:
		for ax in (ax_bottom, ax_top):
			sns.histplot(x=s, bins=bins, stat="density", color=color, alpha=alpha, ax=ax)
			# 去掉子图自身的 y 轴标题，只保留整体居中的一个
			ax.set_ylabel("")
	except Exception:
		plt.close(fig)
		return

	# 计算全局最大密度，确定上半部分的 y 范围
	try:
		y_max = max([p.get_height() for p in ax_bottom.patches] or [0.0])
	except Exception:
		y_max = 0.0
	if y_max <= 0.0 or y_high >= y_max:
		# 如果断轴范围不合理，退回单轴
		plt.close(fig)
		hist_density_simple(
			series=series,
			title=title,
			out_png=out_png,
			xlabel=xlabel,
			xlim=xlim,
			bins=bins,
			clip_quantiles=clip_quantiles,
			alpha=alpha,
			color=color,
			y_label=None
		)
		return

	# 如果用户指定了纵轴上限，则在此基础上裁剪
	if y_upper is not None and y_upper > y_high:
		y_top_max = min(y_upper, y_max * 1.05)
	else:
		y_top_max = y_max * 1.05

	# 设置 y 轴范围：下半部分看 [0, y_low]，上半部分看 [y_high, y_top_max]
	ax_bottom.set_ylim(0.0, y_low)
	ax_top.set_ylim(y_high, y_top_max)

	# 去掉中间的 spine，画断轴斜线
	ax_top.spines["bottom"].set_visible(False)
	ax_bottom.spines["top"].set_visible(False)
	ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

	# 断轴斜线长度（相对坐标），调小一点让视觉上更精细
	d = 0.01
	kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
	ax_top.plot((-d, +d), (-d, +d), **kwargs)
	ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
	kwargs.update(transform=ax_bottom.transAxes)
	ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
	ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

	# 标题和坐标轴标签
	ax_top.set_title(title, fontsize=16)
	# 只在图层级画一个纵坐标标签，居中显示
	fig.text(0.02, 0.5, "Density", rotation="vertical", va="center", ha="center", fontsize=14)
	if xlabel:
		ax_bottom.set_xlabel(xlabel, fontsize=14)
	else:
		ax_bottom.set_xlabel(ax_bottom.get_xlabel(), fontsize=14)

	for ax in (ax_top, ax_bottom):
		ax.tick_params(axis="both", labelsize=12)
		if xlim:
			ax.set_xlim(*xlim)

	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close(fig)


