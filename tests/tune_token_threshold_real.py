"""
Finds the optimal TOKEN_MIN_RATIO using real per-step token counts recorded
by tests/measure_token_baseline.py — the rolling-average ratios computed the
SAME WAY as monitor/detector.py's check_token_explosion.

Unlike a single clean-run-vs-single-verbose-run comparison, this version
requires the candidate ratio to hold up across MULTIPLE independent trials
of each category (measure_token_baseline.py's TRIALS_PER_TASK). A candidate
is only "valid" if:
  - it produces ZERO false positives across every ratio in every clean trial
    (not just the worst clean trial — any single false positive anywhere
    in the clean pool disqualifies it)
  - it catches the spike in EVERY verbose trial (each verbose trial must
    have at least one ratio clearing the candidate — reliably catching the
    spike sometimes isn't good enough)

This is a meaningfully stronger bar than a single-pair comparison: it's
asking "does this threshold generalize across independent runs" rather than
"did this threshold separate the one run I happened to record."

Run with:
    python3 -m tests.tune_token_threshold_real

Requires tests/fixtures/token_baseline_results.json to already exist —
generate it first with:
    python3 -m tests.measure_token_baseline
"""

import json
from collections import defaultdict

RESULTS_PATH = "tests/fixtures/token_baseline_results.json"

CANDIDATES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0]


def load_trials(path: str = RESULTS_PATH) -> dict:
    """Group successful trials by base category (e.g. "clean_buggy_add").

    Falls back to inferring the category from the label prefix if
    base_label isn't present (keeps older single-trial result files usable).
    """
    with open(path) as f:
        raw = json.load(f)

    by_base = defaultdict(list)
    for r in raw:
        if "error" in r:
            continue
        base = r.get("base_label") or r["label"].rsplit("_trial", 1)[0]
        by_base[base].append(r)

    clean = {k: v for k, v in by_base.items() if k.startswith("clean")}
    verbose = {k: v for k, v in by_base.items() if k.startswith("verbose")}

    if not clean or not verbose:
        raise ValueError(
            f"Expected at least one successful 'clean_*' and 'verbose_*' "
            f"trial group in {path}, found: {list(by_base.keys())}"
        )

    # flatten to a single list of trials per side (supports multiple base
    # tasks per category, though the current script only defines one each)
    clean_trials = [t for trials in clean.values() for t in trials]
    verbose_trials = [t for trials in verbose.values() for t in trials]
    return {"clean_trials": clean_trials, "verbose_trials": verbose_trials}


def evaluate_ratio(clean_trials: list, verbose_trials: list, candidate: float) -> dict:
    # a false positive is ANY ratio in ANY clean trial clearing the candidate
    clean_pool = [r for t in clean_trials for r in t["step_vs_rolling_ratios"]]
    false_positives = [r for r in clean_pool if r >= candidate]

    # a verbose trial counts as "caught" if AT LEAST ONE of its ratios
    # clears the candidate; we require ALL verbose trials to be caught
    caught_trials = [
        t for t in verbose_trials
        if any(r >= candidate for r in t["step_vs_rolling_ratios"])
    ]
    missed_trials = [t for t in verbose_trials if t not in caught_trials]

    return {
        "candidate": candidate,
        "false_positives": len(false_positives),
        "verbose_trials_caught": len(caught_trials),
        "verbose_trials_total": len(verbose_trials),
        "missed_trial_labels": [t["label"] for t in missed_trials],
        "valid": len(false_positives) == 0 and len(missed_trials) == 0,
    }


if __name__ == "__main__":
    print(f"Loading real token baseline results from {RESULTS_PATH}...\n")
    data = load_trials()
    clean_trials = data["clean_trials"]
    verbose_trials = data["verbose_trials"]

    clean_pool = [r for t in clean_trials for r in t["step_vs_rolling_ratios"]]
    verbose_maxes = [
        (t["label"], max(t["step_vs_rolling_ratios"]) if t["step_vs_rolling_ratios"] else None)
        for t in verbose_trials
    ]

    print(f"Clean trials: {len(clean_trials)}")
    for t in clean_trials:
        print(f"  {t['label']}: ratios={t['step_vs_rolling_ratios']}  "
              f"max={max(t['step_vs_rolling_ratios']) if t['step_vs_rolling_ratios'] else None}")
    print(f"  Pooled clean ratio ceiling (max across all trials/steps): "
          f"{max(clean_pool) if clean_pool else None}")

    print(f"\nVerbose trials: {len(verbose_trials)}")
    for t in verbose_trials:
        print(f"  {t['label']}: ratios={t['step_vs_rolling_ratios']}  "
              f"max={max(t['step_vs_rolling_ratios']) if t['step_vs_rolling_ratios'] else None}")
    weakest_spike = min((m for _, m in verbose_maxes if m is not None), default=None)
    print(f"  Weakest spike across trials (must still be caught): {weakest_spike}")

    if clean_pool and weakest_spike is not None and weakest_spike <= max(clean_pool):
        print(
            f"\nWARNING: the weakest verbose spike ({weakest_spike}) does not clear "
            f"the pooled clean ceiling ({max(clean_pool)}). No single TOKEN_MIN_RATIO "
            f"can separate these reliably across all trials — either the verbose task "
            f"isn't a strong enough provocation on every trial, or clean-run variance "
            f"is wider than expected. More trials or a more reliably verbose task may "
            f"be needed before trusting a derived threshold."
        )

    print(f"\n{'Ratio':<8}{'FP (pooled clean)':<20}{'Verbose trials caught':<24}{'Valid?':<8}")
    print("-" * 65)
    results = [evaluate_ratio(clean_trials, verbose_trials, c) for c in CANDIDATES]
    for r in results:
        caught = f"{r['verbose_trials_caught']}/{r['verbose_trials_total']}"
        print(f"{r['candidate']:<8}{r['false_positives']:<20}{caught:<24}"
              f"{'yes' if r['valid'] else 'no':<8}")

    valid = [r for r in results if r["valid"]]

    if valid:
        most_sensitive = min(valid, key=lambda r: r["candidate"])

        # The smallest valid candidate sits with ZERO margin against clean
        # variance we haven't sampled yet — it's only as safe as the clean
        # data is complete. The midpoint of the actual observed gap gives
        # real headroom on both sides instead of hugging one edge, at the
        # cost of being slightly less sensitive to borderline spikes.
        clean_ceiling = max(clean_pool) if clean_pool else 0.0
        midpoint = round((clean_ceiling + weakest_spike) / 2, 2) if weakest_spike else most_sensitive["candidate"]

        print(f"\nMost sensitive valid candidate: {most_sensitive['candidate']} "
              f"(catches spikes earliest, but zero margin above the observed "
              f"clean ceiling of {clean_ceiling})")
        print(f"Margin-balanced midpoint: {midpoint} "
              f"(halfway between clean ceiling {clean_ceiling} and weakest spike "
              f"{weakest_spike} — real headroom on both sides)")
        print(f"\nRecommended TOKEN_MIN_RATIO: {midpoint}")
        print(f"  Using the margin-balanced midpoint rather than the most-sensitive "
              f"value, since it isn't safe to assume the clean trials sampled here "
              f"cover the full range of normal variance.")
    else:
        # report how close the best candidate got, rather than silently
        # falling back to an arbitrary number
        closest = min(results, key=lambda r: (r["false_positives"], -r["verbose_trials_caught"]))
        print(f"\nNo candidate in {CANDIDATES} satisfies both constraints across all trials.")
        print(f"Closest: {closest['candidate']} "
              f"(FP={closest['false_positives']}, "
              f"caught={closest['verbose_trials_caught']}/{closest['verbose_trials_total']})")
        print(f"Consider: more trials, a stronger verbose-provoking task, or accepting "
              f"a small false-positive/miss rate at the closest candidate.")

    print(f"\nCurrent value in monitor/scorer.py + monitor/detector.py: TOKEN_MIN_RATIO = 2.75")
