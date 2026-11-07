"""Tests pour l'horloge en ligne (trading_bot.clock)."""

from __future__ import annotations

import time
import urllib.error
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from trading_bot.clock import (
    OnlineClock,
    get_online_now,
    sync_online_clock,
)


def test_online_clock_set_reference_time() -> None:
    """Vérifie que la fixation manuelle de référence (ex: Alpaca) décale correctement l'heure."""
    clock = OnlineClock()
    target_dt = datetime(2026, 11, 5, 20, 55, 0, tzinfo=UTC)
    clock.set_reference_time(target_dt)

    assert clock.is_synced is True
    now = clock.now()
    assert now.tzinfo is UTC
    # Doit être très proche de target_dt (quelques millisecondes écoulées)
    diff = abs((now - target_dt).total_seconds())
    assert diff < 0.1


def test_online_clock_monotonic_progression() -> None:
    """Vérifie que l'heure avance de façon monotone indépendamment de l'horloge système."""
    clock = OnlineClock()
    fixed_time = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    clock.set_reference_time(fixed_time)

    t1 = clock.now()
    time.sleep(0.05)
    t2 = clock.now()

    assert t2 > t1
    diff = (t2 - t1).total_seconds()
    assert 0.04 <= diff <= 0.15


def test_online_clock_offset_calculation() -> None:
    """Vérifie le calcul du décalage avec l'heure locale."""
    clock = OnlineClock()
    # On simule une heure en ligne en avance de 1 heure (3600 s)
    future_online = datetime.now(UTC) + timedelta(hours=1)
    clock.set_reference_time(future_online)

    offset = clock.offset_seconds()
    assert 3590.0 < offset < 3610.0


def test_fetch_online_time_http_header_success() -> None:
    """Vérifie la lecture et conversion du header HTTP Date."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.headers = {"date": "Thu, 05 Nov 2026 21:55:00 GMT"}

    clock = OnlineClock()
    with patch("urllib.request.urlopen", return_value=mock_resp):
        dt = clock._fetch_online_time()

    assert dt is not None
    assert dt == datetime(2026, 11, 5, 21, 55, 0, tzinfo=UTC)


def test_fetch_online_time_http_401_header() -> None:
    """Vérifie que même une réponse HTTP 401 (ex: Alpaca sans auth) permet de lire l'heure."""
    http_err = urllib.error.HTTPError(
        url="https://paper-api.alpaca.markets",
        code=401,
        msg="Unauthorized",
        hdrs={"date": "Thu, 05 Nov 2026 21:55:00 GMT"},  # type: ignore[arg-type]
        fp=None,
    )

    clock = OnlineClock()
    with patch("urllib.request.urlopen", side_effect=http_err):
        dt = clock._fetch_online_time()

    assert dt is not None
    assert dt == datetime(2026, 11, 5, 21, 55, 0, tzinfo=UTC)


def test_fetch_online_time_offline_fallback() -> None:
    """Vérifie le repli sans crash quand toutes les sources réseau échouent."""
    clock = OnlineClock(sources=("https://invalid.test",))
    with patch("urllib.request.urlopen", side_effect=OSError("réseau coupé")):
        dt = clock._fetch_online_time()

    assert dt is None
    # now() doit quand même renvoyer une heure timezone-aware (repli système)
    fallback = clock.now()
    assert fallback.tzinfo is UTC


def test_global_sync_online_clock() -> None:
    """Vérifie le point d'entrée global sync_online_clock."""
    target = datetime(2026, 10, 15, 14, 0, 0, tzinfo=UTC)
    res = sync_online_clock(target)
    assert res is True
    now = get_online_now()
    assert abs((now - target).total_seconds()) < 0.1
