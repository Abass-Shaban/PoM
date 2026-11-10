"""Curated league catalogue (ESPN league slugs).

These are identifiers and branding metadata only — every fixture, score,
standing and statistic shown in the app is fetched from the live provider.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LeagueMeta:
    slug: str
    name: str
    short_name: str
    country: str
    logo: str
    sort_order: int


LEAGUES: dict[str, LeagueMeta] = {
    "eng.1": LeagueMeta(
        "eng.1", "Premier League", "Premier League", "England",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/23.png", 10,
    ),
    "esp.1": LeagueMeta(
        "esp.1", "LaLiga", "LaLiga", "Spain",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/15.png", 20,
    ),
    "ita.1": LeagueMeta(
        "ita.1", "Serie A", "Serie A", "Italy",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/12.png", 30,
    ),
    "ger.1": LeagueMeta(
        "ger.1", "Bundesliga", "Bundesliga", "Germany",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/10.png", 40,
    ),
    "fra.1": LeagueMeta(
        "fra.1", "Ligue 1", "Ligue 1", "France",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/9.png", 50,
    ),
    "uefa.champions": LeagueMeta(
        "uefa.champions", "UEFA Champions League", "Champions League", "Europe",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/2.png", 60,
    ),
    "uefa.europa": LeagueMeta(
        "uefa.europa", "UEFA Europa League", "Europa League", "Europe",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/2310.png", 70,
    ),
    "usa.1": LeagueMeta(
        "usa.1", "Major League Soccer", "MLS", "United States",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/19.png", 80,
    ),
    "ned.1": LeagueMeta(
        "ned.1", "Eredivisie", "Eredivisie", "Netherlands",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/11.png", 90,
    ),
    "por.1": LeagueMeta(
        "por.1", "Primeira Liga", "Liga Portugal", "Portugal",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/14.png", 100,
    ),
    "tur.1": LeagueMeta(
        "tur.1", "Süper Lig", "Süper Lig", "Türkiye",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/18.png", 110,
    ),
    "bra.1": LeagueMeta(
        "bra.1", "Brasileirão Série A", "Brasileirão", "Brazil",
        "https://a.espncdn.com/i/leaguelogos/soccer/500/85.png", 120,
    ),
}

FEATURED_SLUGS: list[str] = list(LEAGUES.keys())
