"""Figures 1-7 of the manuscript (paper/NeuroImage/figures/); 1 and 3-7 from the local results
folder.

Colours: the three tasks take the first three slots of the validated categorical palette
(all-pairs CVD-safe on white); joint estimates and the replication sample use neutral ink.
Scenes and sentences have similar lightness, so each task also has its own marker (and line
style in Fig. 2B), and identity never rests on colour alone, also in grey scale.

    python scripts/paper_figures.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle
from scipy import stats

from n400fp.paths import REPO_ROOT, results_root, task_results_dir

TASKS = ("erpcore", "scene", "sentence")
LABELS = {"erpcore": "Word pairs", "scene": "Scenes", "sentence": "Sentences"}
COLORS = {"erpcore": "#2a78d6", "scene": "#eb6834", "sentence": "#1baf7a"}
MARKERS = {"erpcore": "o", "scene": "s", "sentence": "^"}
LINESTYLES = {"erpcore": "-", "scene": "--", "sentence": "-."}
MINUS = "−"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BASELINE = "B1-W1-N2-C1"
OUT = REPO_ROOT / "paper" / "NeuroImage" / "figures"
FULL_WIDTH = 7.2  # inches, NeuroImage full page width (190 mm)

CATEGORY_SHORT = {
    "general": "general",
    "same direction, size varies": "consistent",
    "task-dependent": "task-dependent",
    "no reliable general effect": "none",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "mathtext.default": "regular",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png")


def mean_ci(x: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of the t-based 95% confidence interval."""
    x = np.asarray(x, dtype=float)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)


def hgrid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def vgrid(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def subject_scores(task: str) -> pd.DataFrame:
    scores = pd.read_csv(task_results_dir(task) / "reports" / "full" / "subject_scores.csv")
    return scores[scores.model == "lr"]


# --- Figure 1: overview ----------------------------------------------------------------------

FIT, MISFIT = "#1a7f4b", "#c23b22"  # labels of the congruent and incongruent examples
PURSUE_BASELINE_TEST = "baseline > 0.5"


def monitor(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """Screen outline with a small stand; (x, y) is the lower left corner of the screen."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.4",
                                fc="white", ec=INK2, lw=0.8))  # fmt: skip
    ax.plot([x + w / 2, x + w / 2], [y, y - 0.6], color=INK2, lw=0.8)
    ax.plot([x + w / 2 - 1.2, x + w / 2 + 1.2], [y - 0.6, y - 0.6], color=INK2, lw=0.8)


def speaker(ax: plt.Axes, x: float, y: float) -> None:
    """Loudspeaker with two sound waves; (x, y) is the centre of the speaker body."""
    body = [(x - 1.0, y - 0.55), (x - 0.3, y - 0.55), (x + 0.5, y - 1.2), (x + 0.5, y + 1.2),
            (x - 0.3, y + 0.55), (x - 1.0, y + 0.55)]  # fmt: skip
    ax.add_patch(Polygon(body, closed=True, fc=INK2, ec=INK2, lw=0.6))
    for r in (1.0, 1.7):
        ax.add_patch(Arc((x + 0.5, y), 2 * r, 2 * r, theta1=-45, theta2=45, color=INK2, lw=0.8))


def example_label(ax: plt.Axes, x: float, y: float, fits: bool) -> None:
    text, color = ("fits", FIT) if fits else ("does not fit", MISFIT)
    ax.text(x, y, text, fontsize=7, color=color, ha="center", va="top", fontweight="bold")


def kitchen(ax: plt.Axes, x: float, y: float, w: float, h: float, item: str) -> None:
    """Schematic kitchen inside a screen: wall cupboards, a counter with a hob, and an object."""
    line = {"color": MUTED, "lw": 0.6}
    ax.add_patch(Rectangle((x + 0.6, y + h - 2.0), w - 1.2, 1.3, fc="none", ec=MUTED, lw=0.6))
    ax.plot([x + w / 2, x + w / 2], [y + h - 2.0, y + h - 0.7], **line)
    ax.plot([x + 0.4, x + w - 0.4], [y + 2.0, y + 2.0], **line)
    ax.add_patch(Rectangle((x + 0.6, y + 0.5), w - 1.2, 1.5, fc="none", ec=MUTED, lw=0.6))
    for dx in (0.25, 0.42):
        ax.add_patch(Ellipse((x + w * dx, y + 2.15), 1.0, 0.3, fc="none", ec=MUTED, lw=0.6))
    ox, oy = x + w * 0.74, y + 3.0
    ax.add_patch(Ellipse((ox, oy), 4.0, 2.0, fc="white", ec=INK, lw=0.6, ls=(0, (2, 1.5))))
    ax.text(ox, oy, item, fontsize=7, color=INK, ha="center", va="center", fontweight="bold")


def task_rows(ax: plt.Axes) -> None:
    """Panel A: one row per task with a congruent and an incongruent example (schematic)."""
    rows = (
        ("erpcore", "Word pairs", "visual,\n40 participants"),
        ("scene", "Scenes", "visual,\n40 participants"),
        ("sentence", "Sentences", "spoken, the\nsame 40"),
    )
    tile_w, tile_h, xs = 10.0, 7.0, (12.5, 25.0)
    screen_y, screen_h = 2.0, 5.0  # screen position within a row
    for (task, name, who), y in zip(rows, (23.0, 12.0, 1.0), strict=True):
        ax.add_patch(Rectangle((0, y), 0.7, tile_h, color=COLORS[task], lw=0))
        ax.text(1.4, y + tile_h, name, fontsize=7.5, fontweight="bold", color=INK, va="top")
        ax.text(1.4, y + tile_h - 1.7, who, fontsize=7, color=INK2, va="top", linespacing=1.25)
        for x, fits in zip(xs, (True, False), strict=True):
            cx = x + tile_w / 2
            if task == "erpcore":
                monitor(ax, x, y + screen_y, tile_w, screen_h)
                target = "butter" if fits else "chair"
                ax.text(cx, y + 5.6, "bread", fontsize=7, color=INK2, ha="center",
                        va="center")  # fmt: skip
                ax.text(cx, y + 3.5, target, fontsize=7, color=INK, ha="center", va="center",
                        fontweight="bold")  # fmt: skip
            elif task == "scene":
                monitor(ax, x, y + screen_y, tile_w, screen_h)
                kitchen(ax, x, y + screen_y, tile_w, screen_h, "pan" if fits else "shoe")
            else:
                speaker(ax, cx - 0.5, y + 6.0)
                ax.text(cx, y + 4.1, "He spread butter", fontsize=7, color=INK2, ha="center",
                        va="center")  # fmt: skip
                ax.text(cx - 0.2, y + 2.6, "on his", fontsize=7, color=INK2, ha="right",
                        va="center")  # fmt: skip
                ax.text(cx, y + 2.6, "bread." if fits else "socks.", fontsize=7, color=INK,
                        ha="left", va="center", fontweight="bold")  # fmt: skip
            example_label(ax, cx, y + 0.9, fits)


def schematic_trial(fs: int = 256) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """A synthetic single trial (1/f noise, alpha and an N400-like negativity) and its band
    envelopes; for illustration only."""
    from scipy.signal import butter, filtfilt, hilbert

    rng = np.random.default_rng(7)
    t = np.arange(-1.0, 1.8, 1 / fs)
    spectrum = np.fft.rfft(rng.standard_normal(t.size))
    freqs = np.fft.rfftfreq(t.size, 1 / fs)
    noise = np.fft.irfft(spectrum / np.maximum(freqs, 1.0), t.size)
    noise *= 2.5 / noise.std()
    alpha = 1.2 * np.sin(2 * np.pi * 10 * t + 0.6)
    n400 = -5.0 * np.exp(-((t - 0.40) ** 2) / (2 * 0.08**2))
    p2 = 2.0 * np.exp(-((t - 0.20) ** 2) / (2 * 0.04**2))
    x = noise + alpha + n400 + p2
    bands = {"θ": (4, 8), "α": (8, 13), "β": (13, 30), "γ": (30, 45)}
    env = {}
    for name, (lo, hi) in bands.items():
        b, a = butter(4, [lo, hi], btype="band", fs=fs)
        env[name] = np.abs(hilbert(filtfilt(b, a, x)))
    keep = (t >= -0.2) & (t <= 0.8)
    return t[keep], x[keep], {k: v[keep] for k, v in env.items()}


def trace(ax: plt.Axes, t: np.ndarray, y: np.ndarray, box: tuple[float, float, float, float],
          lim: tuple[float, float], **kw: object) -> None:  # fmt: skip
    """Draw y(t) into the data rectangle box = (x, y, w, h), scaling t (-0.2..0.8 s) and lim."""
    bx, by, bw, bh = box
    xs = bx + (t + 0.2) / 1.0 * bw
    ys = by + (y - lim[0]) / (lim[1] - lim[0]) * bh
    ax.plot(xs, ys, **kw)


def eeg_head(ax: plt.Axes, x: float, y: float, r: float) -> None:
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK2, lw=0.8))
    ax.add_patch(Polygon([(x - 0.6, y + r - 0.1), (x, y + r + 0.9), (x + 0.6, y + r - 0.1)],
                         closed=False, fc="none", ec=INK2, lw=0.8))  # fmt: skip
    for side in (-1, 1):
        ax.add_patch(Ellipse((x + side * r, y), 0.7, 1.8, fc="white", ec=INK2, lw=0.8))
    for ex, ey in [(0, 0), (-1.6, 0.8), (1.6, 0.8), (-1.6, -1.2), (1.6, -1.2), (0, 2.0),
                   (0, -2.0), (-2.6, -0.2), (2.6, -0.2)]:  # fmt: skip
        ax.add_patch(Circle((x + ex, y + ey), 0.32, fc=INK2, lw=0))


def feature_paths(ax: plt.Axes) -> None:
    """Panel B: single EEG trial, the two families of representations, and the classifier."""
    t, x, env = schematic_trial()
    eeg_head(ax, 44.5, 19.0, 3.6)
    ax.text(44.5, 13.6, "EEG", fontsize=7, color=INK2, ha="center", va="top")
    arrow(ax, (48.8, 19.0), (51.0, 19.0))

    trial = (51.5, 15.0, 15.0, 8.0)
    n400 = Rectangle((trial[0] + 0.5 * trial[2], trial[1] - 0.3), 0.2 * trial[2], trial[3] + 0.6,
                     fc=GRID, lw=0)  # fmt: skip
    ax.add_patch(n400)
    ax.plot([trial[0] + 0.2 * trial[2]] * 2, [trial[1] - 0.3, trial[1] + trial[3] + 0.3],
            color=MUTED, lw=0.6, ls=":")  # fmt: skip
    trace(ax, t, x, trial, (-10, 8), color=INK, lw=0.6)
    ax.text(trial[0] + trial[2] / 2, trial[1] + trial[3] + 1.4, "One trial, one channel",
            fontsize=7, color=INK, ha="center")  # fmt: skip
    ax.text(trial[0] + 0.6 * trial[2], trial[1] - 0.8, "N400", fontsize=7, color=INK2, ha="center",
            va="top")  # fmt: skip
    ax.text(trial[0] + 0.2 * trial[2], trial[1] - 0.8, "0 s", fontsize=7, color=INK2, ha="center",
            va="top")  # fmt: skip

    wave = (71.0, 22.0, 13.5, 5.5)
    power = (71.0, 1.5, 13.5, 8.5)
    arrow(ax, (trial[0] + trial[2] + 0.4, 20.0), (70.3, wave[1] + wave[3] / 2))
    arrow(ax, (trial[0] + trial[2] + 0.4, 18.0), (70.3, power[1] + power[3] / 2))

    bins = np.arange(-0.2, 0.8001, 0.05)
    means = [x[(t >= a) & (t < a + 0.05)].mean() for a in bins[:-1]]
    trace(ax, t, x, wave, (-10, 8), color=AXIS, lw=0.5)
    sel = bins[:-1] >= 0.2
    trace(ax, np.repeat(bins[:-1][sel], 2)[1:], np.repeat(np.array(means)[sel], 2)[:-1], wave,
          (-10, 8), color=INK, lw=1.1)  # fmt: skip
    branch_title(ax, wave, "Waveform", "voltage averaged in 50-ms bins")

    for k, (name, e) in enumerate(env.items()):
        lane = (power[0], power[1] + (3 - k) * power[3] / 4, power[2], power[3] / 4 - 0.3)
        trace(ax, t, e, lane, (0, e.max() * 1.05), color=INK2, lw=0.7)
        ax.text(power[0] + power[2] + 0.4, lane[1] + lane[3] / 2, name, fontsize=7,
                color=INK2, ha="left", va="center")  # fmt: skip
    branch_title(ax, power, "Band power", "envelopes, 4 bands")

    arrow(ax, (86.0, wave[1] + wave[3] / 2), (88.5, 21.0))
    arrow(ax, (86.6, power[1] + power[3] / 2), (88.5, 15.0))
    box(ax, (88.7, 12.5, 11.3, 11.5), "Classifier",
        "fits or not?\nsingle trials,\nwithin each\nparticipant")  # fmt: skip
    note = (
        "Each representation crossed with\n4 time windows × 4 normalizations\n"
        "× 5 channel sets = 640 configurations"
    )
    ax.text(40.5, 10.0, note, fontsize=7, color=INK2, va="top", linespacing=1.3)


def branch_title(ax: plt.Axes, rect: tuple[float, float, float, float], title: str,
                 subtitle: str) -> None:  # fmt: skip
    top = rect[1] + rect[3]
    ax.text(rect[0], top + 3.3, title, fontsize=7.5, fontweight="bold", color=INK, va="bottom")
    ax.text(rect[0], top + 1.0, subtitle, fontsize=7, color=INK2, va="bottom")


def overview_effect_sizes(ax: plt.Axes) -> None:
    """Panel C: generalized eta squared of the four main effects, per task and averaged."""
    dims = ("B", "W", "N", "C")
    names = ("Represen-\ntation", "Time\nwindow", "Normali-\nzation", "Channel\nset")
    ges = {
        task: pd.read_csv(task_results_dir(task) / "reports" / "full" / "rm_anova_lr.csv")
        .set_index("effect")
        .ges
        for task in TASKS
    }
    mean = [np.mean([ges[t][d] for t in TASKS]) for d in dims]
    ax.bar(range(4), mean, width=0.6, color=GRID, ec=AXIS, lw=0.6, zorder=1)
    for k, task in enumerate(TASKS):
        ax.plot(np.arange(4) + (k - 1) * 0.16, [ges[task][d] for d in dims], MARKERS[task],
                color=COLORS[task], ms=3.8, mec="white", mew=0.5, ls="", zorder=2,
                label=LABELS[task])  # fmt: skip
    ax.text(0, mean[0] + 0.012, f"{mean[0]:.3f}", fontsize=7, ha="center", color=INK)
    ax.set_xticks(range(4), names, fontsize=7)
    ax.set_ylabel("Effect size (generalized η²)")
    ax.set_ylim(0, 0.31)
    hgrid(ax)
    ax.legend(loc="upper right", fontsize=7, handletextpad=0.2)


def overview_accuracies() -> list[tuple[str, list[float], str, str]]:
    """Mean accuracy (%) of B2, B1 and B8 at the baseline settings for each sample."""
    configs = ("B2-W1-N2-C1", BASELINE, "B8-W1-N2-C1")
    rows = []
    for task in TASKS:
        wide = subject_scores(task).pivot(index="subject", columns="config", values="bacc")
        rows.append((LABELS[task], [100 * wide[c].mean() for c in configs], COLORS[task],
                     MARKERS[task]))  # fmt: skip
    tests = pd.read_csv(results_root() / "pursue" / "reports" / "replication" / "tests.csv")
    main = tests[(tests.model == "lr") & tests["sample"].str.startswith("main")].set_index("test")
    b1 = 0.5 + main.loc[PURSUE_BASELINE_TEST, "mean"]
    pursue = [b1 - main.loc["R2", "mean"], b1, b1 + main.loc["R1", "mean"]]
    rows.append(("Replication (n = 283)", [100 * v for v in pursue], INK, "D"))
    return rows


def overview_representations(ax: plt.Axes) -> None:
    """Panel D: theta power, waveform, and waveform plus band power, per sample."""
    for name, values, color, marker in overview_accuracies():
        ax.plot(range(3), values, color=color, lw=1.0, marker=marker, ms=4, mec="white",
                mew=0.5, label=name)  # fmt: skip
    ax.axhline(50, color=AXIS, lw=0.8)
    ax.text(1.0, 49.6, "chance", fontsize=7, color=MUTED, ha="center", va="top")
    ax.set_ylim(47.5, 72)
    ax.set_xticks(range(3), ["Theta-band\npower", "Waveform", "Waveform +\nband power"],
                  fontsize=7)  # fmt: skip
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylabel("Balanced accuracy (%)")
    hgrid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)


def figure_overview() -> None:
    fig = plt.figure(figsize=(FULL_WIDTH, 4.9))
    top = fig.add_axes((0.0, 0.47, 1.0, 0.53))
    top.set_xlim(0, 100)
    top.set_ylim(0, 36)
    top.set_aspect("equal", adjustable="datalim")
    top.axis("off")
    top.text(0, 35.5, "A   Three semantic tasks", fontsize=8.5, fontweight="bold", va="top")
    top.text(40.0, 35.5, "B   From a single EEG trial to features", fontsize=8.5,
             fontweight="bold", va="top")  # fmt: skip
    task_rows(top)
    feature_paths(top)

    ax_c = fig.add_axes((0.075, 0.08, 0.30, 0.31))
    ax_d = fig.add_axes((0.50, 0.08, 0.28, 0.31))
    fig.text(0.0, 0.43, "C   Which choice matters?", fontsize=8.5, fontweight="bold")
    fig.text(0.425, 0.43, "D   Waveform versus band power", fontsize=8.5, fontweight="bold")
    overview_effect_sizes(ax_c)
    overview_representations(ax_d)
    save(fig, "fig1_overview")


# --- Figure 2: design ------------------------------------------------------------------------

BOX_FILL = "#f6f6f4"


def box(ax: plt.Axes, xywh: tuple[float, float, float, float], title: str, body: str = "",
        stripe: str | None = None) -> None:  # fmt: skip
    """Rounded box in data units (x, y, width, height) with a bold title and smaller body."""
    x, y, w, h = xywh
    style = "round,pad=0,rounding_size=0.8"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=BOX_FILL, ec=AXIS, lw=0.8))
    if stripe:
        ax.add_patch(Rectangle((x, y + 0.6), 0.9, h - 1.2, color=stripe, lw=0))
    tx = x + (2.0 if stripe else 1.4)
    ax.text(tx, y + h - 1.2, title, fontsize=7.5, fontweight="bold", va="top", color=INK)
    if body:
        ax.text(tx, y + h - 4.0, body, fontsize=7, va="top", color=INK2, linespacing=1.3)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    props = {"arrowstyle": "-|>", "color": INK2, "lw": 0.9, "shrinkA": 0, "shrinkB": 0}
    ax.annotate("", xy=end, xytext=start, arrowprops=props)


def figure1() -> None:
    fig = plt.figure(figsize=(FULL_WIDTH, 3.9))
    ax = fig.add_axes((0, 0, 1, 1))  # the whole figure, so that data units match the text size
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")
    cols = {"data": (0, 20), "pre": (23.5, 21), "feat": (48, 25), "dec": (76.5, 23.5)}
    heads = {"data": "Data", "pre": "Preprocessing", "feat": "Feature processing",
             "dec": "Decoding and inference"}  # fmt: skip
    for key, (x, _) in cols.items():
        ax.text(x, 52, heads[key], fontsize=8.5, fontweight="bold", color=INK, va="top")

    x, w = cols["data"]
    tasks = [
        ("Word pairs", "ERP CORE N400, n = 40\nvisual word pairs,\nrelatedness; 60 per class"),
        ("Scenes", "Draschkow et al., n = 40\nobject consistent or not\nwith scene; 76 per class"),
        ("Sentences", "same 40 participants\nspoken German sentences;\n80 trials per class"),
    ]
    for (title, body), task, y in zip(tasks, TASKS, (38, 27, 16), strict=True):
        box(ax, (x, y, w, 10), title, body, stripe=COLORS[task])
        arrow(ax, (x + w, y + 5), (cols["pre"][0], y + 5))
    pursue = "PURSUE N400, n = 283\n3 sites, ERP CORE paradigm"
    box(ax, (x, 1, w, 9.5), "Replication", pursue, stripe=INK2)

    x, w = cols["pre"]
    path = (
        "example path of Kessler\net al. (2025), close to the\nERP CORE pipeline:\n"
        "ocular and muscle ICA,\n0.1 Hz high-pass, epochs\n−0.4 to 0.8 s, autoreject\n"
        "interpolation; adapted per\ndataset (Section 2.2)\n\n"
        "Band envelopes on the\ncontinuous data: band-pass,\nHilbert, log, baseline"
    )
    box(ax, (x, 16, w, 32), "One fixed path", path)
    arrow(ax, (x + w, 32), (cols["feat"][0], 32))

    x, w = cols["feat"]
    dims = [
        ("Representation (8)", "waveform; theta, alpha, beta,\nlow-gamma envelopes; combinations"),
        ("Time window (4)", "200–800, 300–500, 500–800,\n0–800 ms (50-ms bins)"),
        ("Normalization (4)", "none, per feature, per trial,\nnoise normalization"),
        ("Channels (5)", "all 28, centro-parietal, frontal,\nposterior, symmetrized"),
    ]
    for (title, body), y in zip(dims, (40.5, 32.5, 24.5, 16.5), strict=True):
        box(ax, (x, y, w, 7.5), title, body)
    total = "8 × 4 × 4 × 5 = 640 configurations"
    ax.text(x + w / 2, 14.6, total, fontsize=7, color=INK, ha="center", va="top")
    arrow(ax, (x + w, 43), (cols["dec"][0], 43))

    x, w = cols["dec"]
    decoding = (
        "L2 logistic regression, LDA:\n10-fold CV × 5, shared folds;\n"
        "EEGNet: 5-fold CV × 3 seeds;\nbalanced accuracy"
    )
    box(ax, (x, 36, w, 12), "Decoding per participant", decoding)
    arrow(ax, (x + w / 2, 36), (x + w / 2, 34.5))
    joint = "11 pre-specified contrasts\nequal task weights\ngeneral / consistent / none"
    box(ax, (x, 24.5, w, 10), "Joint test, three tasks", joint)
    arrow(ax, (x + w / 2, 24.5), (x + w / 2, 23))
    tfr = "wavelet power 2–36 Hz, phase;\nconcatenated and stacked\nwith the waveform"
    box(ax, (x, 13, w, 10), "Time–frequency check", tfr)
    replication = "R1: B8 < B1;  R2: B1 > B2\nbaseline settings, one-sided"
    box(ax, (x, 1, w, 9.5), "Pre-registered replication", replication)
    arrow(ax, (cols["data"][1], 5.75), (x, 5.75))
    label = "same preprocessing and decoding, 3 configurations"
    mid = (cols["data"][1] + x) / 2
    ax.text(mid, 6.6, label, fontsize=7, color=INK2, ha="center", va="bottom")
    save(fig, "fig2_design")


# --- Figure 2: decodability ----------------------------------------------------------------


def panel_baseline(ax: plt.Axes) -> None:
    rng = np.random.default_rng(0)
    for i, task in enumerate(TASKS):
        s = subject_scores(task)
        acc = s[s.config == BASELINE].bacc.to_numpy()
        ax.scatter(
            i + rng.uniform(-0.18, 0.18, len(acc)),
            acc,
            s=10,
            marker=MARKERS[task],
            color=COLORS[task],
            alpha=0.55,
            linewidths=0,
        )
        m, h = mean_ci(acc)
        ax.errorbar(i + 0.32, m, yerr=h, fmt="o", color=INK, ms=4, capsize=0, lw=1.2)
    rep = pd.read_csv(results_root() / "pursue" / "reports" / "replication" / "tests.csv")
    row = rep[(rep.model == "lr") & (rep.test == "baseline > 0.5")].iloc[0]
    ax.errorbar(
        3,
        0.5 + row["mean"],
        yerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
        fmt="D",
        color=INK2,
        ms=4,
        capsize=0,
        lw=1.2,
    )
    ax.axhline(0.5, color=AXIS, lw=0.8)
    ax.set_xticks(
        range(4),
        [
            "Word pairs\n(n = 40)",
            "Scenes\n(n = 40)",
            "Sentences\n(n = 40)",
            "Replication\n(n = 283)",
        ],
    )
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("A  Baseline configuration", loc="left", fontweight="bold")
    hgrid(ax)


def panel_timecourse(ax: plt.Axes) -> None:
    for k, task in enumerate(TASKS):
        tc = pd.read_csv(task_results_dir(task) / "reports" / "full" / "timecourse.csv")
        tc = tc[tc.channels == "C1"].sort_values("bin_start")
        t = (tc.bin_start.to_numpy() + 0.025) * 1000
        ax.fill_between(t, tc.ci_low, tc.ci_high, color=COLORS[task], alpha=0.15, lw=0)
        ax.plot(
            t, tc.mean_bacc, color=COLORS[task], ls=LINESTYLES[task], lw=1.5, label=LABELS[task]
        )
        sig = tc.p_tfce.to_numpy() < 0.05
        y = 0.474 - 0.006 * k
        for tt in t[sig]:  # contiguous 50-ms segments form one bar per significant run
            ax.plot([tt - 25, tt + 25], [y, y], color=COLORS[task], lw=2.5, solid_capstyle="butt")
    ax.axhline(0.5, color=AXIS, lw=0.8)
    ax.set_xlim(0, 800)
    ax.set_xlabel("Time after stimulus (ms)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(
        "B  Time-resolved decoding (waveform, all channels)", loc="left", fontweight="bold"
    )
    ax.legend(loc="upper right")
    hgrid(ax)


def figure2() -> None:
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 2.8), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    panel_baseline(a)
    panel_timecourse(b)
    fig.tight_layout()
    save(fig, "fig3_decodability")


# --- Figure 3: marginal means ----------------------------------------------------------------

LEVEL_NAMES = {
    "B": [
        "B1 waveform",
        "B2 theta",
        "B3 alpha",
        "B4 beta",
        "B5 low gamma",
        "B6 four bands",
        "B7 relative",
        "B8 waveform + bands",
    ],
    "W": ["W1 200–800", "W2 300–500", "W3 500–800", "W4 0–800"],
    "N": ["N1 none", "N2 per feature", "N3 per-trial z", "N4 noise norm."],
    "C": ["C1 all 28", "C2 centro-parietal", "C3 frontal", "C4 posterior", "C5 symmetrized"],
}
DIM_TITLES = {"B": "Representation", "W": "Time window (ms)", "N": "Normalization", "C": "Channels"}


def within_ci(wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Means and Cousineau-Morey 95% half-widths of within-participant levels."""
    norm = wide.sub(wide.mean(axis=1), axis=0) + wide.to_numpy().mean()
    n, k = wide.shape
    half = stats.t.ppf(0.975, n - 1) * norm.std(ddof=1) / np.sqrt(n) * np.sqrt(k / (k - 1))
    return wide.mean(), half


def figure3() -> None:
    fig, axes = plt.subplots(
        1, 4, figsize=(FULL_WIDTH, 3.1), sharey=True, gridspec_kw={"width_ratios": [8, 4, 4, 5]}
    )
    for ax, dim in zip(axes, "BWNC", strict=True):
        for k, task in enumerate(TASKS):
            table = pd.read_csv(
                task_results_dir(task) / "reports" / "full" / "marginal_tables" / f"lr_{dim}.csv"
            )
            wide = table.pivot(index="subject", columns=dim, values="acc")
            means, half = within_ci(wide)
            x = np.arange(len(means)) + (k - 1) * 0.22
            ax.errorbar(
                x,
                means,
                yerr=half,
                fmt=MARKERS[task],
                ms=3.5,
                color=COLORS[task],
                lw=1,
                capsize=0,
                label=LABELS[task],
            )
        ax.axhline(0.5, color=AXIS, lw=0.8)
        ax.set_xticks(
            range(len(LEVEL_NAMES[dim])),
            LEVEL_NAMES[dim],
            rotation=40,
            ha="right",
            rotation_mode="anchor",
            fontsize=7.5,
        )
        ax.set_title(DIM_TITLES[dim], loc="left", fontweight="bold")
        hgrid(ax)
    axes[0].set_ylabel("Marginal balanced accuracy")
    axes[0].legend(loc="upper center", fontsize=7)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig4_marginal_means")


# --- Figures 4 and 6: joint tests ------------------------------------------------------------


def forest(
    ax: plt.Axes, joint: pd.DataFrame, labels: list[str], notes: dict[str, str] | None = None
) -> None:
    """Joint estimate with CI (ink diamond) and the three task estimates, in percentage points.
    ``notes`` appends a marker to a category label (the footnotes of Table 3)."""
    notes = notes or {}
    n = len(joint)
    for i, r in enumerate(joint.itertuples()):
        y = n - 1 - i
        yj = y + 0.16  # joint estimate above, the three tasks stacked below it
        ax.plot(
            [100 * r.ci_low, 100 * r.ci_high], [yj, yj], color=INK, lw=1.4, solid_capstyle="butt"
        )
        ax.plot(100 * r.estimate, yj, "D", color=INK, ms=4.5)
        for k, task in enumerate(TASKS):
            ax.plot(
                100 * getattr(r, f"{task}_mean"),
                y - 0.04 - 0.14 * k,
                MARKERS[task],
                ms=3.8,
                color=COLORS[task],
                mec="white",
                mew=0.5,
            )
        ax.text(
            1.01,
            y,
            CATEGORY_SHORT[r.category] + notes.get(r.contrast, ""),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=7,
            color=INK2,
        )
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(range(n), labels[::-1])
    ax.set_ylim(-0.7, n - 0.4)
    ax.set_xlabel("Difference in balanced accuracy (percentage points)")
    vgrid(ax)
    handles = [plt.Line2D([], [], marker="D", color=INK, ls="", ms=5, label="Joint (95% CI)")]
    handles += [
        plt.Line2D([], [], marker=MARKERS[t], color=COLORS[t], ls="", ms=4, label=LABELS[t])
        for t in TASKS
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7)


# The labels of Table 3 of the manuscript.
CONTRAST_LABELS = {
    "P1": "B8 - B1 (add band envelopes)",
    "P2": "B6 - best single band",
    "P3": "B1 - B2 (waveform vs. theta)",
    "P4": "W1 - W2 (add late window)",
    "P5": "W4 - W1 (add 0–200 ms)",
    "P6": "N4 - N2 (noise normalization)",
    "P7": "N3 - N2 (per-trial z-score)",
    "P8": "C1 - C2 (all vs. N400 region)",
    "P9": "C2 - C3 (N400 region vs. frontal)",
    "P10": "C2 - C4 (N400 region vs. posterior)",
    "P11": "(C1 - C5)$_{W3}$ - (C1 - C5)$_{W2}$",
}


def figure4() -> None:
    joint = pd.read_csv(results_root() / "joint" / "joint_contrasts_lr.csv")
    labels = [f"{c}  {CONTRAST_LABELS[c]}".replace(" - ", f" {MINUS} ") for c in joint.contrast]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.9, 3.6))
    forest(ax, joint, labels, notes={"P2": " †", "P6": " ‡"})
    fig.tight_layout()
    save(fig, "fig5_joint_test")


def figure6() -> None:
    joint = pd.read_csv(results_root() / "tfr_sensitivity" / "joint.csv")
    joint = joint.rename(columns={"comparison": "contrast"})
    names = {
        "S1": "T4 - B1 (concatenated)",
        "S2": "T5 - B1 (stacked)",
        "S3": "T1 - B6 (wavelet vs. bands)",
        "S4": "T1 - B1 (wavelet vs. waveform)",
        "S5": "T2 - T1 (up to 7.4 Hz)",
        "S6": "T3 - B1 (phase vs. waveform)",
    }
    # called S1-S6 in the pre-specification; F1-F6 in the paper (S is the supplement prefix)
    labels = [f"F{c[1:]}  {names[c]}".replace(" - ", f" {MINUS} ") for c in joint.contrast]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.75, 2.6))
    forest(ax, joint, labels)
    fig.tight_layout()
    save(fig, "fig7_time_frequency")


# --- Figure 5: replication -------------------------------------------------------------------


def conditional_effect(task: str, a: str, b: str) -> tuple[float, float]:
    """Mean and CI half-width of the participant-level difference a - b (logistic regression)."""
    s = subject_scores(task).pivot(index="subject", columns="config", values="bacc")
    return mean_ci(s[a] - s[b])


def replication_rows(test: str, a: str, b: str) -> list[tuple[str, float, float, float, str, str]]:
    """(label, mean, CI low, CI high, colour, marker) for the sample, the sites and the tasks."""
    rep_dir = results_root() / "pursue" / "reports" / "replication"
    tests, sites = pd.read_csv(rep_dir / "tests.csv"), pd.read_csv(rep_dir / "by_site.csv")
    main = tests[
        (tests.model == "lr") & (tests.test == test) & tests["sample"].str.startswith("main")
    ].iloc[0]
    rows = [("Replication, all (n = 283)", main["mean"], main.ci_low, main.ci_high, INK, "D")]
    for r in sites[(sites.model == "lr") & (sites.test == test)].itertuples():
        rows.append(
            (
                f"  {r.site.split(' (')[0].capitalize()} (n = {r.n})",
                r.mean,
                r.ci_low,
                r.ci_high,
                INK2,
                "o",
            )
        )
    for task in TASKS:
        m, h = conditional_effect(task, a, b)
        rows.append((f"{LABELS[task]} (n = 40)", m, m - h, m + h, COLORS[task], MARKERS[task]))
    return rows


def figure5() -> None:
    specs = [
        ("R1", "B8-W1-N2-C1", BASELINE, f"R1  B8 {MINUS} B1: adding band envelopes"),
        ("R2", BASELINE, "B2-W1-N2-C1", f"R2  B1 {MINUS} B2: waveform vs. theta envelope"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4), sharey=True)
    for ax, (test, a, b, title) in zip(axes, specs, strict=True):
        rows = replication_rows(test, a, b)
        for i, (_, m, lo, hi, color, marker) in enumerate(rows):
            y = len(rows) - 1 - i - (0.5 if i >= 4 else 0)
            ax.plot([100 * lo, 100 * hi], [y, y], color=color, lw=1.4, solid_capstyle="butt")
            ax.plot(100 * m, y, marker, color=color, ms=5 if i == 0 else 4)
        ys = [len(rows) - 1 - i - (0.5 if i >= 4 else 0) for i in range(len(rows))]
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8)
        ax.set_xlabel("Difference in balanced accuracy (percentage points)")
        vgrid(ax)
    fig.tight_layout()
    save(fig, "fig6_replication")


def main() -> None:
    figure_overview()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()


if __name__ == "__main__":
    main()
