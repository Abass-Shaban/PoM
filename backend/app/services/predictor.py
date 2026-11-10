"""Prediction engine.

Statistical core: a Dixon-Coles adjusted bivariate Poisson score model.
Machine learning: a scikit-learn classifier (see ``ml.py``) blended with the
Poisson probabilities when a trained artifact is available.

All inputs come from real stored matches.  The response exposes both model
outputs, the blend weight and the underlying sample sizes so nothing is hidden
behind a black box.
"""
from __future__ import annotations

import math

from .features import context_metrics
from .ml import model_status, predict_from_metrics
from .runtime import get_blend_weight

OUTCOME_KEYS = ("home", "draw", "away")
OUTCOME_LABELS = {"home": "Home win", "draw": "Draw", "away": "Away win"}
GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
TEAM_CORNER_LINES = (3.5, 4.5, 5.5, 6.5)
RHO = -0.045
CAP = 10


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(lam, 1e-6)
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, max(k, 0) + 1))


def score_matrix(home_goals: float, away_goals: float, rho: float = RHO, cap: int = CAP) -> list[list[float]]:
    home_pmf = [poisson_pmf(i, home_goals) for i in range(cap + 1)]
    away_pmf = [poisson_pmf(i, away_goals) for i in range(cap + 1)]
    matrix: list[list[float]] = []
    for x in range(cap + 1):
        row = []
        for y in range(cap + 1):
            tau = 1.0
            if x == 0 and y == 0:
                tau = 1.0 - home_goals * away_goals * rho
            elif x == 0 and y == 1:
                tau = 1.0 + home_goals * rho
            elif x == 1 and y == 0:
                tau = 1.0 + away_goals * rho
            elif x == 1 and y == 1:
                tau = 1.0 - rho
            row.append(home_pmf[x] * away_pmf[y] * max(tau, 0.001))
        matrix.append(row)
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        return matrix
    return [[value / total for value in row] for row in matrix]


def _outcomes(matrix: list[list[float]]) -> dict[str, float]:
    home = draw = away = 0.0
    for x, row in enumerate(matrix):
        for y, value in enumerate(row):
            if x > y:
                home += value
            elif x == y:
                draw += value
            else:
                away += value
    return {"home": home, "draw": draw, "away": away}


def _btts(matrix: list[list[float]]) -> float:
    return sum(
        value
        for x, row in enumerate(matrix)
        for y, value in enumerate(row)
        if x > 0 and y > 0
    )


def _over(matrix: list[list[float]], line: float) -> float:
    return sum(
        value
        for x, row in enumerate(matrix)
        for y, value in enumerate(row)
        if x + y > line
    )


def _correct_scores(matrix: list[list[float]], top: int = 6) -> list[dict]:
    cells = [
        (x, y, value)
        for x, row in enumerate(matrix)
        for y, value in enumerate(row)
    ]
    cells.sort(key=lambda item: item[2], reverse=True)
    return [
        {"score": f"{x}-{y}", "home": x, "away": y, "probability": _round(value)}
        for x, y, value in cells[:top]
    ]


def _first_to_score(home_goals: float, away_goals: float) -> dict:
    total_rate = home_goals + away_goals
    no_goal = math.exp(-total_rate)
    if total_rate <= 0:
        return {"home": 0.0, "away": 0.0, "none": 1.0}
    home_first = (home_goals / total_rate) * (1 - no_goal)
    away_first = (away_goals / total_rate) * (1 - no_goal)
    return {
        "home": _round(home_first),
        "away": _round(away_first),
        "none": _round(no_goal),
    }


def _shrink_strength(raw: float, sample: int, k: float = 4.0) -> float:
    weight = sample / (sample + k) if sample > 0 else 0.0
    return 1.0 + (raw - 1.0) * weight


def _team_strength(profile: dict, baseline: dict, venue: str) -> dict:
    split = profile.get(venue) or {}
    played = split.get("played") or 0
    sample = profile.get("sample_size") or 0
    if venue == "home":
        league_for = baseline.get("home_goals_avg") or 1.45
        league_against = baseline.get("away_goals_avg") or 1.15
    else:
        league_for = baseline.get("away_goals_avg") or 1.15
        league_against = baseline.get("home_goals_avg") or 1.45

    if played >= 2 and split.get("avg_goals_for") is not None:
        attack_raw = (split.get("avg_goals_for") or 0) / max(league_for, 0.2)
        defence_raw = (split.get("avg_goals_against") or 0) / max(league_against, 0.2)
        sample_used = played
        source = "venue"
    else:
        league_mean = ((baseline.get("home_goals_avg") or 1.45) + (baseline.get("away_goals_avg") or 1.15)) / 2
        attack_raw = (profile.get("weighted_goals_for") or profile.get("avg_goals_for") or 1.3) / max(league_mean, 0.2)
        defence_raw = (profile.get("weighted_goals_against") or profile.get("avg_goals_against") or 1.2) / max(league_mean, 0.2)
        sample_used = sample
        source = "overall"

    attack = _clamp(_shrink_strength(attack_raw, sample_used), 0.5, 1.95)
    defence = _clamp(_shrink_strength(defence_raw, sample_used), 0.5, 1.95)
    ppg = profile.get("ppg") or 1.35
    form_factor = _clamp(1.0 + (ppg - 1.35) * 0.05, 0.92, 1.08)
    return {
        "attack": _round(attack, 3),
        "defence": _round(defence, 3),
        "form_factor": _round(form_factor, 3),
        "sample": sample_used,
        "source": source,
        "ppg": _round(ppg, 2),
    }


def _h2h_adjustment(h2h: list[dict], home_id: int) -> tuple[float, float, int]:
    diffs = []
    for game in h2h[:5]:
        hs, as_ = game.get("home_score"), game.get("away_score")
        if hs is None or as_ is None:
            continue
        home_team = (game.get("home") or {}).get("id")
        away_team = (game.get("away") or {}).get("id")
        if home_team == home_id:
            diffs.append(hs - as_)
        elif away_team == home_id:
            diffs.append(as_ - hs)
    if not diffs:
        return 1.0, 1.0, 0
    average = sum(diffs) / len(diffs)
    home_factor = _clamp(1.0 + average * 0.02, 0.93, 1.07)
    away_factor = _clamp(1.0 - average * 0.02, 0.93, 1.07)
    return home_factor, away_factor, len(diffs)


def _corner_model(context: dict, baseline: dict) -> dict:
    home = context.get("home") or {}
    away = context.get("away") or {}
    home_profile = (home.get("profiles") or {}).get("10") or {}
    away_profile = (away.get("profiles") or {}).get("10") or {}
    home_split = home_profile.get("home") or {}
    away_split = away_profile.get("away") or {}
    league_home = baseline.get("home_corners_avg") or 5.4
    league_away = baseline.get("away_corners_avg") or 4.6
    corner_sample = int(baseline.get("corner_sample") or 0)
    home_corner_sample = int(home_profile.get("corner_sample") or 0)
    away_corner_sample = int(away_profile.get("corner_sample") or 0)

    if corner_sample < 5 and home_corner_sample < 3 and away_corner_sample < 3:
        return {
            "available": False,
            "reason": "The provider has not returned enough corner statistics for these teams yet.",
            "expected_total": None,
            "expected_home": None,
            "expected_away": None,
            "over_under": [],
            "team_over_under": [],
            "most_corners": None,
            "distribution": [],
            "sample_size": corner_sample,
        }

    home_attack_raw = (
        (home_split.get("avg_corners_for") or home_profile.get("avg_corners_for") or league_home)
        / max(league_home, 1.0)
    )
    away_defence_raw = (
        (away_split.get("avg_corners_against") or away_profile.get("avg_corners_against") or league_home)
        / max(league_home, 1.0)
    )
    away_attack_raw = (
        (away_split.get("avg_corners_for") or away_profile.get("avg_corners_for") or league_away)
        / max(league_away, 1.0)
    )
    home_defence_raw = (
        (home_split.get("avg_corners_against") or home_profile.get("avg_corners_against") or league_away)
        / max(league_away, 1.0)
    )

    home_attack = _clamp(_shrink_strength(home_attack_raw, max(home_corner_sample, 2), 3), 0.55, 1.8)
    away_defence = _clamp(_shrink_strength(away_defence_raw, max(away_corner_sample, 2), 3), 0.55, 1.8)
    away_attack = _clamp(_shrink_strength(away_attack_raw, max(away_corner_sample, 2), 3), 0.55, 1.8)
    home_defence = _clamp(_shrink_strength(home_defence_raw, max(home_corner_sample, 2), 3), 0.55, 1.8)

    expected_home = _clamp(league_home * home_attack * away_defence, 1.5, 12.0)
    expected_away = _clamp(league_away * away_attack * home_defence, 1.5, 12.0)
    expected_total = expected_home + expected_away

    over_under = []
    for line in CORNER_LINES:
        over = 1 - poisson_cdf(int(math.floor(line)), expected_total)
        over_under.append(
            {
                "line": line,
                "over": _round(over),
                "under": _round(1 - over),
            }
        )

    team_over_under = []
    for label, mean in (("home", expected_home), ("away", expected_away)):
        for line in TEAM_CORNER_LINES:
            over = 1 - poisson_cdf(int(math.floor(line)), mean)
            team_over_under.append(
                {
                    "team": label,
                    "line": line,
                    "over": _round(over),
                    "under": _round(1 - over),
                }
            )

    home_pmf = [poisson_pmf(i, expected_home) for i in range(0, 21)]
    away_pmf = [poisson_pmf(i, expected_away) for i in range(0, 21)]
    home_more = sum(
        home_pmf[i] * away_pmf[j] for i in range(21) for j in range(21) if i > j
    )
    away_more = sum(
        home_pmf[i] * away_pmf[j] for i in range(21) for j in range(21) if i < j
    )
    equal = max(0.0, 1.0 - home_more - away_more)
    total_pmf = [poisson_pmf(i, expected_total) for i in range(0, 21)]
    distribution = [
        {"corners": i, "probability": _round(total_pmf[i])} for i in range(0, 21)
    ]

    return {
        "available": True,
        "reason": None,
        "expected_home": _round(expected_home, 2),
        "expected_away": _round(expected_away, 2),
        "expected_total": _round(expected_total, 2),
        "over_under": over_under,
        "team_over_under": team_over_under,
        "most_corners": {
            "home": _round(home_more),
            "draw": _round(equal),
            "away": _round(away_more),
        },
        "distribution": distribution,
        "sample_size": corner_sample,
        "team_samples": {"home": home_corner_sample, "away": away_corner_sample},
        "model": "independent Poisson (recency-weighted corner rates)",
    }


def _confidence(probabilities: dict[str, float]) -> dict:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_key, top_value = ordered[0]
    margin = top_value - ordered[1][1]
    if top_value >= 0.60 and margin >= 0.12:
        label = "High confidence"
    elif top_value >= 0.46:
        label = "Moderate confidence"
    else:
        label = "Low confidence"
    return {
        "level": label,
        "top": top_key,
        "probability": _round(top_value),
        "margin": _round(margin),
    }


def _market_comparison(context: dict, probabilities: dict[str, float]) -> dict | None:
    odds = (context.get("summary") or {}).get("odds") or {}
    implied = odds.get("implied") or {}
    if not implied:
        return None
    edges = []
    for key in OUTCOME_KEYS:
        market = implied.get(key)
        if market is None:
            continue
        model = probabilities[key]
        edges.append(
            {
                "outcome": key,
                "label": OUTCOME_LABELS[key],
                "model": _round(model),
                "market": _round(market),
                "edge": _round(model - market),
            }
        )
    edges.sort(key=lambda item: item["edge"] or 0, reverse=True)
    return {
        "provider": odds.get("provider"),
        "implied": implied,
        "over_under": odds.get("over_under"),
        "edges": edges,
        "note": "Market probabilities are vig-free implied probabilities and are shown for comparison only.",
    }


def _insights(context: dict, prediction: dict) -> list[str]:
    home = context.get("home") or {}
    away = context.get("away") or {}
    home_team = (home.get("team") or {}).get("name") or "Home"
    away_team = (away.get("team") or {}).get("name") or "Away"
    h10 = (home.get("profiles") or {}).get("10") or {}
    a10 = (away.get("profiles") or {}).get("10") or {}
    probabilities = prediction["probabilities"]
    expected = prediction["expected"]
    insights: list[str] = []

    top_key = prediction["most_likely"]["outcome"]
    top_label = {"home": home_team, "draw": "a draw", "away": away_team}[top_key]
    insights.append(
        f"The blended model gives {top_label} a {_pct(probabilities[top_key])} probability "
        f"({prediction['most_likely']['confidence']})."
    )
    if h10.get("sample_size"):
        insights.append(
            f"{home_team}: {h10['wins']}W-{h10['draws']}D-{h10['losses']}L in the last "
            f"{h10['sample_size']}, scoring {h10['avg_goals_for']:.2f} and conceding "
            f"{h10['avg_goals_against']:.2f} per match."
        )
    if a10.get("sample_size"):
        insights.append(
            f"{away_team}: {a10['wins']}W-{a10['draws']}D-{a10['losses']}L in the last "
            f"{a10['sample_size']}, scoring {a10['avg_goals_for']:.2f} and conceding "
            f"{a10['avg_goals_against']:.2f} per match."
        )
    btts_home = h10.get("btts_rate")
    btts_away = a10.get("btts_rate")
    if btts_home is not None and btts_away is not None:
        insights.append(
            f"BTTS landed in {_pct(btts_home)} of {home_team}'s and {_pct(btts_away)} of "
            f"{away_team}'s recent matches; the model prices it at {_pct(prediction['btts']['yes'])}."
        )
    insights.append(
        f"Expected goals: {home_team} {expected['home_goals']:.2f} — {away_team} "
        f"{expected['away_goals']:.2f} (total {expected['total_goals']:.2f})."
    )
    corners = prediction.get("corners") or {}
    if corners.get("available"):
        insights.append(
            f"Corner projection: {corners['expected_total']:.1f} total "
            f"({home_team} {corners['expected_home']:.1f}, {away_team} {corners['expected_away']:.1f}) "
            f"from {corners.get('sample_size', 0)} league matches with corner data."
        )
    h2h = context.get("h2h") or []
    if h2h:
        goals = [
            (g.get("home_score") or 0) + (g.get("away_score") or 0)
            for g in h2h[:5]
            if g.get("home_score") is not None
        ]
        if goals:
            insights.append(
                f"The last {len(goals)} head-to-head meetings averaged "
                f"{sum(goals) / len(goals):.2f} goals."
            )
    market = prediction.get("market") or {}
    if market.get("edges"):
        best = market["edges"][0]
        if best["edge"] and best["edge"] > 0.02:
            insights.append(
                f"Model vs market: {best['label']} is {best['edge'] * 100:.1f} percentage "
                f"points above the vig-free market probability."
            )
    if prediction["data_quality"]["warnings"]:
        insights.append(prediction["data_quality"]["warnings"][0])
    return insights[:8]


def predict_match(context: dict) -> dict:
    fixture = context.get("fixture") or {}
    home = context.get("home") or {}
    away = context.get("away") or {}
    home_team = home.get("team") or {}
    away_team = away.get("team") or {}
    baseline = context.get("baseline") or {}
    home_profile = (home.get("profiles") or {}).get("10") or {}
    away_profile = (away.get("profiles") or {}).get("10") or {}

    home_strength = _team_strength(home_profile, baseline, "home")
    away_strength = _team_strength(away_profile, baseline, "away")
    h2h_home_factor, h2h_away_factor, h2h_sample = _h2h_adjustment(
        context.get("h2h") or [], home_team.get("id") or 0
    )

    home_goals = (
        (baseline.get("home_goals_avg") or 1.45)
        * home_strength["attack"]
        * away_strength["defence"]
        * home_strength["form_factor"]
        * h2h_home_factor
    )
    away_goals = (
        (baseline.get("away_goals_avg") or 1.15)
        * away_strength["attack"]
        * home_strength["defence"]
        * away_strength["form_factor"]
        * h2h_away_factor
    )
    home_goals = _clamp(home_goals, 0.2, 4.5)
    away_goals = _clamp(away_goals, 0.2, 4.5)

    matrix = score_matrix(home_goals, away_goals)
    poisson = _outcomes(matrix)
    ml = predict_from_metrics(context_metrics(context))
    blend_weight = get_blend_weight() if ml else 0.0
    if ml:
        final = {
            key: (1 - blend_weight) * poisson[key] + blend_weight * ml["probabilities"][key]
            for key in OUTCOME_KEYS
        }
    else:
        final = dict(poisson)
    total = sum(final.values())
    final = {key: value / total for key, value in final.items()}
    poisson_total = sum(poisson.values()) or 1
    poisson = {key: value / poisson_total for key, value in poisson.items()}

    btts_yes = _btts(matrix)
    over_under = [
        {
            "line": line,
            "over": _round(_over(matrix, line)),
            "under": _round(1 - _over(matrix, line)),
        }
        for line in GOAL_LINES
    ]
    most_likely_key = max(final, key=final.get)
    confidence = _confidence(final)
    corners = _corner_model(context, baseline)
    market = _market_comparison(context, final)

    prediction = {
        "fixture_id": fixture.get("id"),
        "generated_at": context.get("generated_at"),
        "is_pre_match": fixture.get("status") == "scheduled",
        "match_status": fixture.get("status"),
        "teams": {
            "home": home_team,
            "away": away_team,
        },
        "model": {
            "version": (ml or {}).get("version") or "poisson-dc",
            "statistical_model": "Dixon-Coles adjusted Poisson",
            "ml": {
                "available": bool(ml),
                "version": (ml or {}).get("version"),
                "algorithm": (ml or {}).get("algorithm"),
                "trained_at": (ml or {}).get("trained_at"),
                "blend_weight": _round(blend_weight, 2) if ml else 0.0,
            },
            "poisson_probabilities": {k: _round(v) for k, v in poisson.items()},
            "ml_probabilities": ({k: _round(v) for k, v in ml["probabilities"].items()} if ml else None),
            "rho": RHO,
        },
        "probabilities": {k: _round(v) for k, v in final.items()},
        "most_likely": {
            "outcome": most_likely_key,
            "label": OUTCOME_LABELS[most_likely_key],
            "team": home_team.get("name") if most_likely_key == "home" else (away_team.get("name") if most_likely_key == "away" else None),
            "probability": _round(final[most_likely_key]),
            "confidence": confidence["level"],
            "margin": confidence["margin"],
        },
        "double_chance": {
            "1X": _round(final["home"] + final["draw"]),
            "12": _round(final["home"] + final["away"]),
            "X2": _round(final["draw"] + final["away"]),
        },
        "btts": {"yes": _round(btts_yes), "no": _round(1 - btts_yes)},
        "goals_over_under": over_under,
        "first_to_score": _first_to_score(home_goals, away_goals),
        "correct_scores": _correct_scores(matrix),
        "expected": {
            "home_goals": _round(home_goals, 2),
            "away_goals": _round(away_goals, 2),
            "total_goals": _round(home_goals + away_goals, 2),
            "home_corners": corners.get("expected_home"),
            "away_corners": corners.get("expected_away"),
            "total_corners": corners.get("expected_total"),
        },
        "corners": corners,
        "market": market,
        "strengths": {
            "home": home_strength,
            "away": away_strength,
            "h2h_sample": h2h_sample,
            "h2h_home_factor": _round(h2h_home_factor, 3),
            "h2h_away_factor": _round(h2h_away_factor, 3),
        },
        "data_quality": context.get("data_quality"),
        "params": {
            "goal_cap": CAP,
            "decay_half_life_days": 180,
            "corner_model": "Poisson",
            "blend": "Poisson + scikit-learn when available",
        },
        "disclaimer": (
            "Probabilities are statistical estimates derived from historical match data, "
            "not guarantees. Football outcomes are uncertain; use responsibly."
        ),
    }
    prediction["insights"] = _insights(context, prediction)
    return prediction
