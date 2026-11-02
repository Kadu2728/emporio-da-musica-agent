from emporio_agent import policies


def names(result):
    return [p["nome"] for p in result["resultados"]]


# ---------------------------------------------------------------- catalog

def test_search_by_category_and_budget(tools):
    result = tools.search_products(category="Violões", max_price=1000)
    assert result["total_encontrado"] > 0
    assert all(float(p["preco_final"].replace("R$ ", "").replace(".", "").replace(",", ".")) <= 1000 for p in result["resultados"])
    assert "Giannini GF-3D Dreadnought Sunburst" not in names(result)  # out of stock


def test_search_is_accent_and_plural_insensitive(tools):
    assert names(tools.search_products(query="violoes nylon eletricos")) == names(tools.search_products(query="violão nylon elétrico"))


def test_search_category_without_products(tools):
    result = tools.search_products(category="Instrumentos de Sopro (Metais)")
    assert result["resultados"] == [] and "Não há nenhum produto" in result["observacao"]


def test_product_details_tolerates_formatting(tools):
    result = tools.get_product_details("takamine gd-20")
    assert result["produtos"][0]["nome"] == "Takamine GD20 Dreadnought Natural"
    assert result["produtos"][0]["preco_final"] == "R$ 2.199,00"


def test_out_of_stock_product_offers_alternatives(tools):
    product = tools.get_product_details("Giannini GF-3D")["produtos"][0]
    assert product["disponibilidade"] == "esgotado no momento"
    assert product["alternativas_disponiveis"]


def test_discontinued_product(tools):
    product = tools.get_product_details("Shelby SN-7C")["produtos"][0]
    assert "descontinuado" in product["disponibilidade"]


def test_active_promotion_is_shown_with_original_price_and_no_pix_stacking(tools):
    product = tools.get_product_details("Crafter HT-100")["produtos"][0]
    assert product["preco_tabela"] == "R$ 2.399,00"
    assert product["promocao_ativa"]["desconto_percentual"] == 18
    assert product["preco_final"] == "R$ 1.967,18"
    assert "não é cumulativo" in product["pix"]


def test_inactive_promotions_are_ignored(tools):
    product = tools.get_product_details("Martin D-28 Dreadnought")["produtos"][0]
    assert "promocao_ativa" not in product  # promo 6 and 23 exist but are inactive
    promos = tools.list_active_promotions()["promocoes_ativas"]
    assert {p["id"] for p in promos} == {90, 94, 121, 127}


# ---------------------------------------------------------------- orders / privacy

def test_order_lookup_requires_matching_identity(tools):
    ok = tools.lookup_orders("(67) 99812-3456", order_id=4)
    assert ok["encontrado"] and ok["pedidos"][0]["cliente_primeiro_nome"] == "Lucas"
    # right order, wrong customer -> generic answer, no data leak
    wrong = tools.lookup_orders("ana.carolina@x.com", order_id=4)
    assert not wrong["encontrado"] and "pedidos" not in wrong
    other_customer = tools.lookup_orders("anacarol.ferreira@coldmail.com", order_id=4)
    assert not other_customer["encontrado"]


def test_phone_formats_are_normalised(tools):
    for phone in ("67998123456", "+55 67 99812-3456", "99812-3456"):
        assert tools.lookup_orders(phone)["encontrado"]
    assert not tools.lookup_orders("3456")["encontrado"]  # too short to identify someone


def test_regret_window_expired(tools):
    result = tools.check_return_eligibility(4, "lucas.mendes@jmail.com", "arrependimento")
    assert result["elegivel"] is False and result["dias_desde_recebimento"] > 7


def test_defect_inside_legal_warranty(tools):
    result = tools.check_return_eligibility(7, "leticia.goncalves@x.com", "defeito")
    assert result["encontrado"] is False  # wrong e-mail
    result = tools.check_return_eligibility(7, "(67) 99789-0123", "defeito")
    assert result["elegivel"] is True and "garantia legal" in result["regra"]


def test_returns_for_orders_not_yet_shipped(tools):
    result = tools.check_return_eligibility(13, "(67) 99456-7890", "arrependimento")
    assert result["elegivel"] is None and "ainda não foi enviado" in result["motivo"]


# ---------------------------------------------------------------- policies

def test_policy_manual_has_all_sections():
    sections = policies.load_sections()
    assert sorted(sections) == list(range(1, 11))
    assert "Rua 14 de Maio, 3200" in sections[1].text


def test_policy_errata_applied():
    text = " ".join(s.text for s in policies.load_sections().values())
    assert "empóriodamusica" not in text and "3321-4500" not in text


def test_get_store_policy_by_topic(tools):
    result = tools.get_store_policy(["pagamento", "horario"])
    assert len(result["trechos_do_manual"]) == 2
    assert "12x sem juros" in result["trechos_do_manual"][1]


def test_tool_call_with_bad_arguments_returns_error(tools):
    assert "erro" in tools.call("get_product_details", {"nome": "x"})
    assert "erro" in tools.call("nope", {})
