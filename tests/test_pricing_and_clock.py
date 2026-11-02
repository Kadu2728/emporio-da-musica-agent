from datetime import datetime

from emporio_agent import clock
from emporio_agent.pricing import brl, max_installments


def test_brl_format():
    assert brl(1234.5) == "R$ 1.234,50"
    assert brl(11499) == "R$ 11.499,00"


def test_installments_follow_policy_tiers():
    assert max_installments(2199) == (12, 183.25)  # >= R$100 per installment
    assert max_installments(599.9) == (6, 99.98)  # 7x would be R$ 85.70 < R$ 100
    assert max_installments(150) == (3, 50.0)  # up to 3x with min R$ 50
    assert max_installments(90) == (1, 90.0)  # 2x would be R$ 45 < 50
    assert max_installments(800)[0] == 8  # 8x = R$100


def test_store_open_and_closed():
    tz = clock.CAMPO_GRANDE_TZ
    assert "ABERTA" in clock.store_status(datetime(2026, 3, 24, 10, 30, tzinfo=tz))
    saturday_afternoon = clock.store_status(datetime(2026, 3, 28, 14, 0, tzinfo=tz))
    assert "FECHADA" in saturday_afternoon and "segunda-feira" in saturday_afternoon
    finados = clock.store_status(datetime(2026, 11, 2, 11, 0, tzinfo=tz))
    assert "Finados" in finados and "terça-feira" in finados


def test_good_friday_is_a_holiday():
    assert clock.holiday_name(datetime(2026, 4, 3).date()) == "Sexta-feira Santa"
