"""Deterministic pricing rules from the policy manual (sections 3 and 6.2).

Money maths is done in code, never by the LLM: the model only receives final numbers to present.
"""

from __future__ import annotations

from .store import Product, Promotion

PIX_DISCOUNT = 0.05
COMBINED_PAYMENT_THRESHOLD = 2000.00

# (min_installments, max_installments, minimum value per installment) — policy 3.1.
# Assumption: "até 3x ... sem valor mínimo (exceto abaixo de R$ 50,00)" means each installment must
# be at least R$ 50. Section 3.1 is more specific than the R$ 100 note in the table of section 3,
# so it takes precedence (documented in the README).
INSTALLMENT_RULES = [(1, 3, 50.00), (4, 6, 80.00), (7, 12, 100.00)]
MAX_INSTALLMENTS = 12


def _min_installment(n: int) -> float:
    for low, high, minimum in INSTALLMENT_RULES:
        if low <= n <= high:
            return minimum
    raise ValueError(n)


def max_installments(amount: float) -> tuple[int, float]:
    """Largest interest-free installment count allowed for `amount`, and the installment value."""
    for n in range(MAX_INSTALLMENTS, 1, -1):
        if amount / n >= _min_installment(n):
            return n, round(amount / n, 2)
    return 1, round(amount, 2)


def brl(value: float) -> str:
    """Brazilian currency format: 1234.5 -> 'R$ 1.234,50'."""
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def quote(product: Product, promotion: Promotion | None) -> dict:
    """Everything the agent needs to present a price transparently."""
    list_price = product.price
    result: dict = {"preco_tabela": brl(list_price)}

    if promotion:
        final = round(list_price * (1 - promotion.discount_percent / 100), 2)
        result["promocao_ativa"] = {
            "nome": promotion.name,
            "desconto_percentual": promotion.discount_percent,
            "preco_promocional": brl(final),
        }
        # Policy 6.2: PIX discount does not stack on promotional prices.
        result["pix"] = f"{brl(final)} (desconto PIX não é cumulativo com promoção)"
    else:
        final = list_price
        result["pix"] = f"{brl(round(final * (1 - PIX_DISCOUNT), 2))} (5% de desconto no PIX)"

    n, value = max_installments(final)
    result["preco_final"] = brl(final)
    result["cartao_credito"] = (
        f"até {n}x sem juros de {brl(value)}" if n > 1 else "à vista (valor abaixo do mínimo para parcelar)"
    )
    result["pode_combinar_formas_de_pagamento"] = final > COMBINED_PAYMENT_THRESHOLD
    result["_valor_final_numerico"] = final
    return result
