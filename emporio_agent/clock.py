"""Current time in Campo Grande and whether the store is open (policy section 2).

"Are you open now?" is a recurring question and a classic LLM failure (models don't know the
time), so opening status is computed here and injected into the system prompt on every turn.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone

# Campo Grande/MS is UTC-4 all year (no DST since 2019). A fixed offset avoids depending on the
# IANA tz database, which is not shipped with Python on Windows.
CAMPO_GRANDE_TZ = timezone(timedelta(hours=-4), "America/Campo_Grande")

OPENING_HOURS = {  # weekday() -> (open, close); Sunday closed
    0: (time(9), time(18)),
    1: (time(9), time(18)),
    2: (time(9), time(18)),
    3: (time(9), time(18)),
    4: (time(9), time(18)),
    5: (time(9), time(13)),
}

WEEKDAYS_PT = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]

# Fixed-date holidays: national + Mato Grosso do Sul (11/10) + Campo Grande (26/08).
# Assumption: the manual only says "Domingo e Feriados: Fechado"; the real list should come
# from the store's calendar.
FIXED_HOLIDAYS = {
    (1, 1): "Confraternização Universal",
    (4, 21): "Tiradentes",
    (5, 1): "Dia do Trabalho",
    (8, 26): "Aniversário de Campo Grande",
    (9, 7): "Independência do Brasil",
    (10, 11): "Criação do Estado de Mato Grosso do Sul",
    (10, 12): "Nossa Senhora Aparecida",
    (11, 2): "Finados",
    (11, 15): "Proclamação da República",
    (11, 20): "Dia da Consciência Negra",
    (12, 25): "Natal",
}


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def holiday_name(day: date) -> str | None:
    if (day.month, day.day) in FIXED_HOLIDAYS:
        return FIXED_HOLIDAYS[(day.month, day.day)]
    easter = _easter(day.year)
    if day == easter - timedelta(days=2):
        return "Sexta-feira Santa"
    if day == easter + timedelta(days=60):
        return "Corpus Christi"
    return None


def now() -> datetime:
    """Current time in Campo Grande. EMPORIO_NOW freezes the clock for demos and tests."""
    frozen = os.getenv("EMPORIO_NOW")
    if frozen:
        return datetime.fromisoformat(frozen).replace(tzinfo=CAMPO_GRANDE_TZ)
    return datetime.now(CAMPO_GRANDE_TZ)


def today() -> date:
    return now().date()


def _hours_for(day: date) -> tuple[time, time] | None:
    if holiday_name(day):
        return None
    return OPENING_HOURS.get(day.weekday())


def next_opening(moment: datetime) -> datetime:
    day = moment.date()
    for offset in range(0, 15):
        candidate = day + timedelta(days=offset)
        hours = _hours_for(candidate)
        if not hours:
            continue
        opening = datetime.combine(candidate, hours[0], tzinfo=moment.tzinfo)
        if opening > moment:
            return opening
    raise RuntimeError("no opening found in the next two weeks")


def store_status(moment: datetime | None = None) -> str:
    """One-line, prompt-ready description of the current time and opening status."""
    moment = moment or now()
    day = moment.date()
    stamp = f"{WEEKDAYS_PT[day.weekday()]}, {day:%d/%m/%Y}, {moment:%H:%M} (horário de Campo Grande/MS)"
    hours = _hours_for(day)
    holiday = holiday_name(day)

    if hours and hours[0] <= moment.time() < hours[1]:
        return f"Agora é {stamp}. A loja está ABERTA (fecha hoje às {hours[1]:%H:%M})."

    reason = f"feriado: {holiday}" if holiday else "fora do horário de funcionamento"
    nxt = next_opening(moment)
    when = "hoje" if nxt.date() == day else f"{WEEKDAYS_PT[nxt.weekday()]} ({nxt:%d/%m})"
    return (
        f"Agora é {stamp}. A loja está FECHADA ({reason}). "
        f"Reabre {when} às {nxt:%H:%M}. Mensagens recebidas agora serão vistas pela equipe na reabertura."
    )
