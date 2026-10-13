# xrd_snr.py
#
# Signal-to-noise ratio (SNR) of X-ray powder diffraction peaks.
#
#   One pattern  (FILE_B = None): SNR of every peak in the pattern.
#   Two patterns (FILE_B set):    SNR of the same reflections measured on two instruments.
#
# For every peak two values are calculated:
#   height SNR = net peak height / noise
#   area SNR   = net peak area / uncertainty of that area
# The noise is the standard deviation of the baseline next to the peak.
#
# Requirements: numpy, matplotlib

import sys

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ settings

FILE_A = "pattern_A.xy"     # first pattern: 2-theta (degrees) and intensity
ANODE_A = "Cu"              # X-ray tube: Cu, Co, Mo, Fe, Cr or Ag
NAME_A = "Instrument A"
TIME_A = None               # total measuring time in seconds, or None

FILE_B = None               # second pattern for a comparison, or None for one pattern only
ANODE_B = "Co"
NAME_B = "Instrument B"
TIME_B = None

MIN_SNR = 5                 # peaks below this height SNR are ignored
D_TOLERANCE = 0.003         # relative d-spacing difference allowed when matching reflections (0.3 %)
OUTPUT_FIGURE = "xrd_snr.png"

# K-alpha1 and K-alpha2 wavelengths in angstrom (Holzer et al., Phys. Rev. A 56, 4554, 1997)
WAVELENGTHS = {
    "Cu": (1.540593, 1.544414),
    "Co": (1.788965, 1.792850),
    "Mo": (0.709317, 0.713607),
    "Fe": (1.936042, 1.939980),
    "Cr": (2.289760, 2.293663),
    "Ag": (0.559422, 0.563813),
}
KA2_RATIO = 0.5             # intensity of K-alpha2 relative to K-alpha1


# ------------------------------------------------------------------ reading data


def load_xy(filename):
    """Read the first two numbers of every line. Lines that are not numbers (headers) are skipped."""
    two_theta = []
    intensity = []
    with open(filename, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.replace(",", " ").replace(";", " ").split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            two_theta.append(x)
            intensity.append(y)
    if len(two_theta) < 50:
        raise SystemExit(f"{filename}: only {len(two_theta)} data lines found.")
    order = np.argsort(two_theta)
    return np.array(two_theta)[order], np.array(intensity)[order]


def d_spacing(two_theta, wavelength):
    """Bragg's law: d = wavelength / (2 sin theta)."""
    return wavelength / (2 * np.sin(np.radians(two_theta / 2)))


def ka2_position(two_theta, wavelengths):
    """2-theta of the K-alpha2 line belonging to a K-alpha1 peak at two_theta."""
    lam1, lam2 = wavelengths
    s = np.sin(np.radians(two_theta / 2)) * lam2 / lam1
    return 2 * np.degrees(np.arcsin(np.minimum(s, 1.0)))


# ------------------------------------------------------------------ finding peaks


def moving_average(y, n):
    return np.convolve(y, np.ones(n) / n, mode="same")


def running_median(y, half_width):
    """Median of y[i - half_width : i + half_width + 1] for every point."""
    result = np.empty(len(y))
    for i in range(len(y)):
        result[i] = np.median(y[max(0, i - half_width):i + half_width + 1])
    return result


def find_peaks(two_theta, intensity, wavelengths):
    """Return a list of peaks, each as a dict with index, two_theta and fwhm.

    A point is a peak when the smoothed pattern is highest there within +-3 points and rises
    more than MIN_SNR times the noise above the median of the pattern within +-1 degree.
    (The median is a stable baseline; the lowest point would itself be a noise dip.)
    """
    step = np.median(np.diff(two_theta))
    smooth = moving_average(intensity, 3)
    window = max(3, int(round(1.0 / step)))
    level = running_median(intensity, window)
    # rough noise: typical distance from that median (the median ignores the peaks)
    rough_noise = 1.4826 * np.median(np.abs(intensity - level))

    peaks = []
    for i in range(3, len(smooth) - 3):
        if smooth[i] < smooth[i - 3:i + 4].max():
            continue
        rise = smooth[i] - level[i]
        if rise < MIN_SNR * rough_noise:
            continue
        new = {"index": i, "two_theta": two_theta[i], "fwhm": peak_width(two_theta, smooth, i, step)}
        # a noisy peak top can have two maxima a few points apart: closer than half a width
        # means the same peak, so keep the higher of the two
        if peaks and new["two_theta"] - peaks[-1]["two_theta"] < 0.5 * max(new["fwhm"], peaks[-1]["fwhm"]):
            if smooth[i] > smooth[peaks[-1]["index"]]:
                peaks[-1] = new
            continue
        peaks.append(new)

    # where K-alpha1 and K-alpha2 are resolved, the K-alpha2 line looks like a separate weaker peak
    real_peaks = []
    for p in peaks:
        is_ka2 = False
        for q in peaks:
            if q is p:
                continue
            distance = abs(p["two_theta"] - ka2_position(q["two_theta"], wavelengths))
            if distance < max(0.5 * p["fwhm"], 2 * step) and smooth[p["index"]] < 0.8 * smooth[q["index"]]:
                is_ka2 = True
        if not is_ka2:
            real_peaks.append(p)
    return real_peaks


def peak_width(two_theta, y, i, step):
    """Full width at half maximum, measured by walking down both sides of the peak."""
    base = min(y[max(0, i - 50):i + 51])
    half = base + (y[i] - base) / 2
    left = i
    while left > 0 and y[left] > half:
        left -= 1
    right = i
    while right < len(y) - 1 and y[right] > half:
        right += 1
    return max(two_theta[right] - two_theta[left], 3 * step)


# ------------------------------------------------------------------ baseline and noise


def baseline_points(two_theta, peaks, wavelengths):
    """True for points that belong to no peak.

    Each peak occupies from 2.5 widths below its K-alpha1 position to 2.5 widths above its
    K-alpha2 position. Everything else is baseline. Deciding this for the whole pattern first
    keeps peak tails out of the noise, also where peaks are crowded.
    """
    is_baseline = np.ones(len(two_theta), dtype=bool)
    for p in peaks:
        start = p["two_theta"] - 2.5 * p["fwhm"]
        end = ka2_position(p["two_theta"], wavelengths) + 2.5 * p["fwhm"]
        is_baseline[(two_theta >= start) & (two_theta <= end)] = False
    return is_baseline


def local_baseline(two_theta, intensity, is_baseline, center):
    """Straight line through the baseline points near a peak, and the noise around that line."""
    # 100 points give the noise to about 7 %; widen the window until there are that many
    for half_window in (1.5, 3.0, 6.0, 12.0, 1000.0):
        near = is_baseline & (np.abs(two_theta - center) <= half_window)
        if near.sum() >= 100:
            break
    if near.sum() < 10:
        return None
    x = two_theta[near]
    y = intensity[near]
    slope, offset = np.polyfit(x, y, 1)
    residual = y - (slope * x + offset)

    # remove leftover weak peaks that were too small to be detected (one 3-sigma clip)
    sigma = np.std(residual)
    keep = residual < 3 * sigma
    slope, offset = np.polyfit(x[keep], y[keep], 1)
    residual = y[keep] - (slope * x[keep] + offset)
    noise = np.std(residual, ddof=2)
    return slope, offset, noise


def neighbour_correlation(two_theta, intensity, is_baseline):
    """Correlation between the noise of neighbouring points, from all baseline points of the pattern.

    Independent counts give 0. Data integrated from a 2D detector image often gives 0.2-0.4.
    It matters for the area: the uncertainty of a sum of N correlated points is
    noise * sqrt(N * (1 + 2 * correlation)).

    Only differences between baseline points are used, because they cancel the slow background:
      differences of neighbours        spread^2 = 2 noise^2 (1 - correlation)
      differences two points apart     spread^2 = 2 noise^2
    so correlation = 1 - (spread1 / spread2)^2. Medians make it insensitive to leftover peak tails.
    """
    step1 = []
    step2 = []
    for i in range(len(intensity) - 2):
        if is_baseline[i] and is_baseline[i + 1] and is_baseline[i + 2]:
            step1.append(intensity[i + 1] - intensity[i])
            step2.append(intensity[i + 2] - intensity[i])
    if len(step1) < 50:
        return 0.0
    spread1 = np.median(np.abs(step1))
    spread2 = np.median(np.abs(step2))
    if spread2 == 0:
        return 0.0
    return min(max(1 - (spread1 / spread2) ** 2, 0.0), 0.9)


# ------------------------------------------------------------------ SNR of the peaks


TOP_FRACTION = 0.7


def peak_height(two_theta, net, peak):
    """Height of the peak top above the baseline, and the index of the highest point.

    The data points rarely hit the exact top, and averaging points cuts the top off. Instead a
    parabola is fitted to the logarithm of the points near the top (above 70 % of the highest
    point, where every peak shape is close to a parabola) and its maximum is taken.
    At least 3 points are always used. If the fit fails, the highest point is used.
    """
    near = np.nonzero(np.abs(two_theta - peak["two_theta"]) <= peak["fwhm"])[0]
    top = near[np.argmax(net[near])]
    limit = TOP_FRACTION * net[top]
    left = top - 1
    while left - 1 >= near[0] and net[left - 1] > limit:
        left -= 1
    right = top + 1
    while right + 1 <= near[-1] and net[right + 1] > limit:
        right += 1
    x = two_theta[left:right + 1] - two_theta[top]
    y = net[left:right + 1]
    if len(x) >= 3 and np.all(y > 0):
        a, b, c = np.polyfit(x, np.log(y), 2, w=y)   # weights: the log of small values is noisy
        if a < 0 and abs(b / (2 * a)) <= x.max() - x.min():
            return np.exp(c - b * b / (4 * a)), top
    return net[top], top


def half_height_width(two_theta, net, top, height):
    """Full width at half of the given height, with linear interpolation between data points."""
    half = height / 2
    left = top
    while left > 0 and net[left - 1] > half:
        left -= 1
    right = top
    while right < len(net) - 1 and net[right + 1] > half:
        right += 1
    # interpolate where the profile crosses the half height on each side
    x_left, x_right = two_theta[left], two_theta[right]
    if left > 0:
        x_left = np.interp(half, [net[left - 1], net[left]], [two_theta[left - 1], two_theta[left]])
    if right < len(net) - 1:
        x_right = np.interp(half, [net[right + 1], net[right]], [two_theta[right + 1], two_theta[right]])
    return x_right - x_left


def measure_pattern(filename, anode):
    """Find the peaks of one pattern and calculate their SNR."""
    wavelengths = WAVELENGTHS[anode]
    two_theta, intensity = load_xy(filename)
    peaks = find_peaks(two_theta, intensity, wavelengths)
    is_baseline = baseline_points(two_theta, peaks, wavelengths)
    correlation = neighbour_correlation(two_theta, intensity, is_baseline)
    print(f"{filename}: {len(two_theta)} points, noise correlation between neighbouring points {correlation:.2f}")

    results = []
    for p in peaks:
        baseline = local_baseline(two_theta, intensity, is_baseline, p["two_theta"])
        if baseline is None or baseline[2] <= 0:
            continue
        slope, offset, noise = baseline
        net = intensity - (slope * two_theta + offset)
        height, top = peak_height(two_theta, net, p)
        width = half_height_width(two_theta, net, top, height)

        # area: everything from 1.5 widths below K-alpha1 to 1.5 widths above K-alpha2
        start = two_theta[top] - 1.5 * width
        end = ka2_position(two_theta[top], wavelengths) + 1.5 * width
        inside = (two_theta >= start) & (two_theta <= end)
        area = net[inside].sum()
        area_error = noise * np.sqrt(inside.sum() * (1 + 2 * correlation))

        height_snr = height / noise
        if height_snr < MIN_SNR:
            continue
        results.append({
            "two_theta": p["two_theta"],
            "d": d_spacing(p["two_theta"], wavelengths[0]),
            "height": height,
            "noise": noise,
            "height_snr": height_snr,
            "area_snr": area / area_error,
        })
    return two_theta, intensity, results


def match_reflections(peaks_a, peaks_b):
    """Pair peaks with the same d-spacing. Closest pairs first; every peak is used once."""
    candidates = []
    for i, a in enumerate(peaks_a):
        for j, b in enumerate(peaks_b):
            difference = abs(a["d"] - b["d"]) / a["d"]
            if difference <= D_TOLERANCE:
                candidates.append((difference, i, j))
    candidates.sort()

    used_a = set()
    used_b = set()
    pairs = []
    for difference, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((peaks_a[i], peaks_b[j]))
    pairs.sort(key=lambda pair: pair[0]["d"])
    return pairs


def median_with_range(values):
    """Median and its 95 % range from resampling the values (bootstrap)."""
    values = np.array(values)
    rng = np.random.default_rng(0)
    medians = [np.median(rng.choice(values, len(values))) for _ in range(2000)]
    return np.median(values), np.percentile(medians, 2.5), np.percentile(medians, 97.5)


# ------------------------------------------------------------------ the two modes


def single_pattern():
    two_theta, intensity, peaks = measure_pattern(FILE_A, ANODE_A)
    if not peaks:
        raise SystemExit("No peaks found. Try a lower MIN_SNR.")

    print(f"{NAME_A}: {len(peaks)} peaks with height SNR >= {MIN_SNR}\n")
    print("  2theta     d (A)     height    noise   height SNR  area SNR")
    for p in peaks:
        print(f"{p['two_theta']:8.3f}  {p['d']:8.4f}  {p['height']:9.4g}  {p['noise']:7.3g}"
              f"  {p['height_snr']:10.1f}  {p['area_snr']:8.1f}")

    height_snr = [p["height_snr"] for p in peaks]
    area_snr = [p["area_snr"] for p in peaks]
    print(f"\nMedian height SNR: {np.median(height_snr):.1f}")
    print(f"Median area SNR:   {np.median(area_snr):.1f}")
    if TIME_A:
        print(f"Median height SNR per square root of a second: {np.median(height_snr) / np.sqrt(TIME_A):.3f}")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
    top.plot(two_theta, intensity, lw=0.8)
    for p in peaks:
        i = np.argmin(np.abs(two_theta - p["two_theta"]))
        top.annotate(f"{p['height_snr']:.0f}", (p["two_theta"], intensity[i]),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
    top.set_xlabel("2θ (degrees)")
    top.set_ylabel("Intensity")
    top.set_title(f"{NAME_A}: peaks labelled with height SNR")
    one_over_d = [1 / p["d"] for p in peaks]
    bottom.plot(one_over_d, height_snr, "o", label="height SNR")
    bottom.plot(one_over_d, area_snr, "s", label="area SNR")
    bottom.set_yscale("log")
    bottom.set_xlabel("1/d (1/Å)")
    bottom.set_ylabel("SNR")
    bottom.legend()
    save_figure(fig)


def two_patterns():
    _, _, peaks_a = measure_pattern(FILE_A, ANODE_A)
    _, _, peaks_b = measure_pattern(FILE_B, ANODE_B)
    pairs = match_reflections(peaks_a, peaks_b)
    if not pairs:
        raise SystemExit("No reflections found in both patterns. Check ANODE_A and ANODE_B.")

    print(f"Peaks: {NAME_A} {len(peaks_a)}, {NAME_B} {len(peaks_b)}, found in both: {len(pairs)}\n")
    print("              height SNR                area SNR")
    print("   d (A)      A       B     A/B       A       B     A/B")
    height_ratio = []
    area_ratio = []
    for a, b in pairs:
        h = a["height_snr"] / b["height_snr"]
        height_ratio.append(h)
        line = f"{a['d']:8.4f} {a['height_snr']:7.1f} {b['height_snr']:7.1f} {h:6.2f}"
        if a["area_snr"] > 0 and b["area_snr"] > 0:
            r = a["area_snr"] / b["area_snr"]
            area_ratio.append(r)
            line += f"  {a['area_snr']:7.1f} {b['area_snr']:7.1f} {r:6.2f}"
        print(line)

    print()
    report("Height SNR", height_ratio)
    report("Area SNR", area_ratio)
    if TIME_A and TIME_B:
        # SNR grows with the square root of measuring time, so compare SNR / sqrt(time)
        factor = np.sqrt(TIME_B / TIME_A)
        print(f"\nPer square root of a second ({NAME_A} {TIME_A:g} s, {NAME_B} {TIME_B:g} s):")
        report("Height SNR", [r * factor for r in height_ratio])
        report("Area SNR", [r * factor for r in area_ratio])
    else:
        print("\nSet TIME_A and TIME_B to compare per square root of measuring time.")

    one_over_d = [1 / a["d"] for a, b in pairs]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    for axis, key, title in ((left, "height_snr", "Height SNR"), (right, "area_snr", "Area SNR")):
        snr_a = [a[key] for a, b in pairs]
        snr_b = [b[key] for a, b in pairs]
        axis.plot(one_over_d, snr_a, "o", label=NAME_A)
        axis.plot(one_over_d, snr_b, "s", label=NAME_B)
        axis.vlines(one_over_d, snr_b, snr_a, color="gray", lw=0.5)
        axis.set_yscale("log")
        axis.set_xlabel("1/d (1/Å)")
        axis.set_ylabel(title)
        axis.legend()
    left.set_title(f"Median ratio A/B = {np.median(height_ratio):.2f}")
    right.set_title(f"Median ratio A/B = {np.median(area_ratio):.2f}")
    save_figure(fig)


def report(label, ratios):
    if not ratios:
        print(f"{label}: no usable peaks")
        return
    median, low, high = median_with_range(ratios)
    better = sum(1 for r in ratios if r > 1)
    print(f"{label}: median A/B = {median:.2f}  (95 % range {low:.2f} to {high:.2f}),"
          f" A higher on {better} of {len(ratios)} reflections")


def save_figure(fig):
    for axis in fig.axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=200)
    print(f"\nFigure saved as {OUTPUT_FIGURE}")
    plt.show()


if __name__ == "__main__":
    # the Windows console cannot print characters such as α or Å unless told to use UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if FILE_B is None:
        single_pattern()
    else:
        two_patterns()
