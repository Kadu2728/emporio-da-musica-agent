"""Tools the agent can call (function calling).

Principle: *facts and rules live in code, language lives in the LLM.* Every tool returns
structured, already-computed data (final prices, eligibility windows, verified order data), so the
model never does arithmetic, date maths or access control on its own.

Each tool is a plain Python function + a JSON-schema spec. The LLM adapters translate the specs
to each provider's format.
"""

from __future__ import annotations

import difflib
import json
import secrets
from dataclasses import dataclass
from typing import Any, Callable

from . import VAR_DIR, clock, policies
from .pricing import brl, quote
from .store import ORDER_STATUS_LABELS, Order, Product, StoreData, payment_label
from .text import compact, normalize, tokens

GENERIC_NOT_FOUND = (
    "Não encontrei nenhum pedido com esses dados. Peça ao cliente para conferir o número do pedido "
    "e o e-mail ou telefone usado na compra. Nunca diga se o pedido existe para outra pessoa."
)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]


class StoreTools:
    def __init__(self, store: StoreData | None = None):
        self.store = store or StoreData()

    # ------------------------------------------------------------ helpers

    def _product_view(self, p: Product, detailed: bool = False) -> dict:
        view = {
            "id": p.id,
            "nome": p.name,
            "categoria": p.category,
            "disponibilidade": p.availability,
            **{k: v for k, v in quote(p, self.store.active_promotion(p.id)).items() if not k.startswith("_")},
        }
        if detailed:
            view["descricao"] = p.description
            view["especificacoes"] = p.specs
        return view

    def _final_price(self, p: Product) -> float:
        return quote(p, self.store.active_promotion(p.id))["_valor_final_numerico"]

    def _alternatives(self, p: Product, limit: int = 3) -> list[dict]:
        """Similar sellable products: same category, closest price first (policy 7.3)."""
        candidates = [
            q for q in self.store.products.values() if q.id != p.id and q.category_id == p.category_id and q.available
        ]
        candidates.sort(key=lambda q: abs(self._final_price(q) - p.price))
        return [self._product_view(q) for q in candidates[:limit]]

    def _verified_order(self, order_id: int, email_or_phone: str) -> Order | None:
        customer_ids = {c.id for c in self.store.find_customers(email_or_phone)}
        order = self.store.orders.get(int(order_id)) if str(order_id).strip().isdigit() else None
        if not order or order.customer_id not in customer_ids:
            return None
        return order

    def _order_view(self, order: Order) -> dict:
        customer = self.store.customers[order.customer_id]
        view = {
            "pedido": order.id,
            "cliente_primeiro_nome": customer.first_name,
            "data_do_pedido": f"{order.order_date:%d/%m/%Y}",
            "status": order.status,
            "status_explicado": ORDER_STATUS_LABELS.get(order.status, order.status),
            "valor_total": brl(order.total),
            "pagamento": payment_label(order.payment_method),
            "itens": [
                {
                    "produto": self.store.products[i.product_id].name if i.product_id in self.store.products else f"produto {i.product_id}",
                    "quantidade": i.quantity,
                }
                for i in order.items
            ],
        }
        if order.tracking_code:
            view["codigo_rastreamento"] = order.tracking_code
            view["rastreio_onde"] = "site dos Correios ou da Jadlog"
        if order.estimated_delivery:
            label = "data_de_entrega_prevista" if order.status != "delivered" else "data_de_entrega_registrada"
            view[label] = f"{order.estimated_delivery:%d/%m/%Y}"
            if order.status == "shipped" and order.estimated_delivery < clock.today():
                view["alerta"] = "Previsão de entrega já passou e o pedido ainda consta como enviado: ofereça abrir um chamado (escalate_to_human)."
        if order.notes:
            view["observacao"] = order.notes
        return view

    # ------------------------------------------------------------ tools

    def search_products(
        self,
        query: str = "",
        category: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        only_available: bool = True,
        limit: int = 8,
    ) -> dict:
        products = list(self.store.products.values())
        if category:
            cat = self.store.category_by_name(category)
            if not cat:
                return {"erro": f"Categoria '{category}' não existe.", "categorias": [c.name for c in self.store.categories.values()]}
            products = [p for p in products if p.category_id == cat.id]
            if not products:
                return {
                    "resultados": [],
                    "observacao": f"Não há nenhum produto cadastrado na categoria {cat.name} no momento.",
                }

        q_tokens = tokens(query)
        partial = False
        if q_tokens:
            def score(p: Product) -> int:
                haystack = set(tokens(f"{p.name} {p.category} {p.description} {' '.join(map(str, p.specs.values()))}"))
                name_compact = compact(p.name)
                return sum(1 for t in q_tokens if t in haystack or (len(t) >= 3 and t in name_compact))

            scored = [(score(p), p) for p in products]
            full = [p for s, p in scored if s == len(q_tokens)]
            if full:
                products = full
            else:  # no product matches every term: fall back to the best partial matches
                best = max((s for s, _ in scored), default=0)
                products = [p for s, p in scored if s == best and s > 0]
                partial = True

        if only_available:
            products = [p for p in products if p.available]
        if min_price is not None:
            products = [p for p in products if self._final_price(p) >= float(min_price)]
        if max_price is not None:
            products = [p for p in products if self._final_price(p) <= float(max_price)]

        products.sort(key=self._final_price)
        result: dict = {
            "total_encontrado": len(products),
            "resultados": [self._product_view(p) for p in products[: max(1, min(int(limit), 15))]],
        }
        if partial:
            result["observacao"] = "Nenhum produto atende a todos os termos; estes são os mais próximos."
        if not products:
            result["observacao"] = "Nenhum produto encontrado com esses filtros."
        return result

    def get_product_details(self, product: str) -> dict:
        """Exact id, then all-tokens name match, then fuzzy match (typos like 'takamine gd 20')."""
        ref = str(product).strip()
        products = self.store.products
        matches: list[Product] = []

        if ref.isdigit() and int(ref) in products:
            matches = [products[int(ref)]]
        if not matches:
            q_tokens, q_compact = tokens(ref), compact(ref)
            for p in products.values():
                name_tokens, name_compact = set(tokens(p.name)), compact(p.name)
                if q_compact and q_compact in name_compact:
                    matches.append(p)
                elif q_tokens and all(t in name_tokens or t in name_compact for t in q_tokens):
                    matches.append(p)
        if not matches:
            names = {normalize(p.name): p for p in products.values()}
            close = difflib.get_close_matches(normalize(ref), list(names), n=3, cutoff=0.55)
            matches = [names[n] for n in close]
        if not matches:
            return {"encontrado": False, "observacao": f"Nenhum produto parecido com '{ref}' no catálogo."}

        if len(matches) > 3:
            return {
                "encontrado": True,
                "ambiguo": True,
                "observacao": "Vários produtos correspondem; peça ao cliente para especificar.",
                "opcoes": [{"id": p.id, "nome": p.name, "preco_final": self._product_view(p)["preco_final"]} for p in matches[:10]],
            }

        out = []
        for p in matches:
            view = self._product_view(p, detailed=True)
            if not p.available:
                view["alternativas_disponiveis"] = self._alternatives(p)
            out.append(view)
        return {"encontrado": True, "produtos": out}

    def list_active_promotions(self) -> dict:
        promos = [p for p in self.store.promotions if p.is_active and p.product_id in self.store.products]
        items = []
        for promo in promos:
            product = self.store.products[promo.product_id]
            if self.store.active_promotion(product.id) is promo:
                items.append(self._product_view(product))
        return {
            "promocoes_ativas": items,
            "regras": "Promoções não são cumulativas e o desconto PIX não se aplica sobre preço promocional. Sujeitas ao estoque, sem reserva de preço.",
        }

    def lookup_orders(self, email_or_phone: str, order_id: int | None = None) -> dict:
        customers = self.store.find_customers(email_or_phone)
        if not customers:
            return {"encontrado": False, "observacao": GENERIC_NOT_FOUND}
        if order_id is not None:
            order = self._verified_order(order_id, email_or_phone)
            if not order:
                return {"encontrado": False, "observacao": GENERIC_NOT_FOUND}
            return {"encontrado": True, "pedidos": [self._order_view(order)]}
        if len(customers) > 1:  # shared identifier: listing would mix different people's orders
            return {
                "encontrado": False,
                "observacao": "Este contato não identifica um único cliente. Peça o número do pedido (ou o telefone) para continuar.",
            }
        orders = sorted((o for o in self.store.orders.values() if o.customer_id == customers[0].id), key=lambda o: o.order_date, reverse=True)
        if not orders:
            return {"encontrado": False, "observacao": "Cliente identificado, mas sem pedidos registrados."}
        return {"encontrado": True, "pedidos": [self._order_view(o) for o in orders]}

    def check_return_eligibility(self, order_id: int, email_or_phone: str, reason: str) -> dict:
        """Applies policy sections 4 and 8 to a verified order with deterministic date maths.

        Assumptions (README): every order in the system is an online purchase (they all ship); the
        delivery date is `estimated_delivery` because the dataset has no actual delivery date.
        """
        order = self._verified_order(order_id, email_or_phone)
        if not order:
            return {"encontrado": False, "observacao": GENERIC_NOT_FOUND}

        today = clock.today()
        base = {"pedido": order.id, "status": ORDER_STATUS_LABELS.get(order.status, order.status), "hoje": f"{today:%d/%m/%Y}"}
        caveats = [
            "Não são elegíveis: instrumentos personalizados/com ajuste sob encomenda, itens de liquidação com aviso de 'venda final' e boquilhas de sopro.",
        ]

        if order.status == "cancelled":
            return {**base, "elegivel": False, "motivo": "O pedido já está cancelado; não há o que devolver.", "observacao": order.notes}
        if order.status in ("pending", "confirmed"):
            return {
                **base,
                "elegivel": None,
                "motivo": "O pedido ainda não foi enviado, então troca/devolução não se aplica. Um pedido de cancelamento precisa ser tratado pela equipe.",
                "proximo_passo": "Oferecer encaminhar para a equipe (escalate_to_human) para solicitar o cancelamento.",
            }
        if order.status == "shipped":
            return {
                **base,
                "elegivel": None,
                "motivo": "O pedido ainda está a caminho. Os prazos de arrependimento e de troca por preferência começam a contar a partir do recebimento.",
                "proximo_passo": "Se o produto chegar avariado, recusar o recebimento e contatar a loja (seção 5.2).",
            }

        received = order.estimated_delivery or order.order_date
        since_receipt = (today - received).days
        since_purchase = (today - order.order_date).days
        base.update({
            "data_da_compra": f"{order.order_date:%d/%m/%Y}",
            "data_de_recebimento_considerada": f"{received:%d/%m/%Y}",
            "dias_desde_recebimento": since_receipt,
            "dias_desde_compra": since_purchase,
        })
        reason = normalize(reason).replace(" ", "_")

        if reason.startswith("arrependimento"):
            ok = since_receipt <= 7
            return {
                **base,
                "regra": "Direito de arrependimento: até 7 dias corridos após o recebimento (seção 4.1).",
                "elegivel": ok,
                "condicoes": [
                    "Embalagem original, sem sinais de uso, com todos os acessórios e manuais.",
                    "Reembolso na mesma forma de pagamento em até 10 dias úteis.",
                    "Frete de devolução por conta da loja.",
                ] if ok else [],
                "alternativas_se_inelegivel": None if ok else [
                    "Se houver defeito de fabricação, a garantia pode se aplicar (pergunte ao cliente).",
                ],
                "ressalvas": caveats,
            }

        if reason.startswith("troca_preferencia") or reason.startswith("preferencia"):
            ok = since_receipt <= 7
            return {
                **base,
                "regra": "Troca por preferência (cor, modelo, tamanho): até 7 dias, sujeita à disponibilidade do produto desejado (seção 4.3). Prazo contado a partir do recebimento (suposição).",
                "elegivel": ok,
                "condicoes": ["Produto em perfeito estado e na embalagem original.", "Diferenças de valor são cobradas ou reembolsadas."] if ok else [],
                "ressalvas": caveats,
            }

        if reason.startswith("defeito"):
            if since_purchase <= 30:
                path, ok = "Troca direta na loja: defeito de fabricação em até 30 dias corridos após a compra (seção 4.2).", True
            elif since_receipt <= 90:
                path, ok = ("Prazo de troca direta (30 dias) expirou, mas o produto está dentro da garantia legal de 90 dias "
                            "a partir do recebimento (seção 8.1). A loja pode intermediar o acionamento da garantia."), True
            else:
                path, ok = ("Prazo de troca direta e garantia legal expirados. Pode haver garantia do fabricante (6 meses a 2 anos, "
                            "conforme o certificado do produto); a loja pode intermediar (seções 4.2 e 8.2)."), None
            return {
                **base,
                "regra": path,
                "elegivel": ok,
                "nao_cobre": "Mau uso, quedas, umidade excessiva, modificações por terceiros, desgaste natural e danos apenas estéticos (seções 4.2 e 8.3).",
                "proximo_passo": "Registrar o caso com escalate_to_human para a equipe avaliar o produto.",
                "ressalvas": caveats,
            }

        return {"erro": "reason deve ser 'arrependimento', 'defeito' ou 'troca_preferencia'."}

    def get_store_policy(self, topics: list[str]) -> dict:
        if isinstance(topics, str):
            topics = [topics]
        unknown = [t for t in topics if t not in policies.TOPICS]
        sections = policies.get_sections(topics)
        result: dict = {"trechos_do_manual": [s.text for s in sections]}
        if unknown:
            result["topicos_invalidos"] = unknown
            result["topicos_validos"] = list(policies.TOPICS)
        return result

    def escalate_to_human(self, reason: str, summary: str, customer_contact: str = "") -> dict:
        """Registers a ticket for the human team (complaints, cancellations, edge cases)."""
        now = clock.now()
        protocol = f"EM{now:%Y%m%d}-{secrets.token_hex(2).upper()}"
        VAR_DIR.mkdir(exist_ok=True)
        with (VAR_DIR / "tickets.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "protocolo": protocol, "criado_em": now.isoformat(), "motivo": reason,
                "resumo": summary, "contato": customer_contact,
            }, ensure_ascii=False) + "\n")
        return {
            "protocolo": protocol,
            "prazo_de_retorno": "até 24 horas úteis",
            "observacao": "Chamado registrado para a equipe da loja.",
        }

    # ------------------------------------------------------------ registry

    def specs(self) -> list[Tool]:
        categories = [c.name for c in self.store.categories.values()]
        topic_help = "; ".join(f"{k}: {v[1]}" for k, v in policies.TOPICS.items())
        return [
            Tool(
                "search_products",
                "Search the instrument catalog. Use for any question about which products exist, availability, "
                "price ranges or recommendations ('violões até R$1000', 'tem guitarra Fender?'). Returns final prices "
                "already including active promotions, PIX price and installment options.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Free-text terms: brand, model, type, feature (e.g. 'nylon', 'elétrico', 'Yamaha'). Can be empty when filtering only by category/price."},
                        "category": {"type": "string", "enum": categories, "description": "Optional category filter."},
                        "min_price": {"type": "number", "description": "Minimum final price in BRL."},
                        "max_price": {"type": "number", "description": "Maximum final price in BRL."},
                        "only_available": {"type": "boolean", "description": "Default true: only sellable items in stock. Set false to also see sold-out, discontinued or coming-soon items."},
                        "limit": {"type": "integer", "description": "Max results (default 8, max 15)."},
                    },
                },
                self.search_products,
            ),
            Tool(
                "get_product_details",
                "Get price, promotion, stock and specs of ONE specific product by name or id (e.g. 'Takamine GD20'). "
                "Tolerates typos. If the product is unavailable it also returns similar available alternatives.",
                {
                    "type": "object",
                    "properties": {"product": {"type": "string", "description": "Product name, model or id."}},
                    "required": ["product"],
                },
                self.get_product_details,
            ),
            Tool(
                "list_active_promotions",
                "List promotions that are active right now, with original price, discount and final price. "
                "Use when the customer asks about promotions, offers, discounts, Black Friday, etc.",
                {"type": "object", "properties": {}},
                self.list_active_promotions,
            ),
            Tool(
                "lookup_orders",
                "Look up a customer's order status, items, tracking code and delivery date. REQUIRES the e-mail or "
                "phone used in the purchase (identity check). Ask the customer for it before calling.",
                {
                    "type": "object",
                    "properties": {
                        "email_or_phone": {"type": "string", "description": "E-mail or phone registered in the purchase."},
                        "order_id": {"type": "integer", "description": "Order number, if the customer provided it."},
                    },
                    "required": ["email_or_phone"],
                },
                self.lookup_orders,
            ),
            Tool(
                "check_return_eligibility",
                "Decide if an order can be returned or exchanged under store policy, computing deadlines from the order "
                "dates. Use for 'quero devolver', 'me arrependi', 'veio com defeito', 'quero trocar'. Requires order "
                "number and the e-mail/phone used in the purchase.",
                {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "integer"},
                        "email_or_phone": {"type": "string"},
                        "reason": {"type": "string", "enum": ["arrependimento", "defeito", "troca_preferencia"],
                                   "description": "arrependimento = regret/withdrawal; defeito = manufacturing defect; troca_preferencia = wants another color/model."},
                    },
                    "required": ["order_id", "email_or_phone", "reason"],
                },
                self.check_return_eligibility,
            ),
            Tool(
                "get_store_policy",
                "Read the official store policy manual. Use for institutional questions: address, contact, opening "
                f"hours, payment, shipping, returns, warranty, promotions rules, privacy. Topics: {topic_help}.",
                {
                    "type": "object",
                    "properties": {
                        "topics": {"type": "array", "items": {"type": "string", "enum": list(policies.TOPICS)}, "description": "One or more topics."},
                    },
                    "required": ["topics"],
                },
                self.get_store_policy,
            ),
            Tool(
                "escalate_to_human",
                "Open a ticket for the human team. Use for complaints, cancellation requests, damaged/defective "
                "products, late deliveries, or anything you cannot solve. Returns a protocol number.",
                {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "enum": ["reclamacao", "cancelamento", "defeito", "atraso_entrega", "outro"]},
                        "summary": {"type": "string", "description": "Short summary of the case for the team, in Portuguese."},
                        "customer_contact": {"type": "string", "description": "Customer name/e-mail/phone if known."},
                    },
                    "required": ["reason", "summary"],
                },
                self.escalate_to_human,
            ),
        ]

    def call(self, name: str, arguments: dict) -> dict:
        tool = next((t for t in self.specs() if t.name == name), None)
        if not tool:
            return {"erro": f"Ferramenta desconhecida: {name}"}
        try:
            return tool.fn(**(arguments or {}))
        except TypeError as exc:  # wrong/missing arguments from the model: let it retry
            return {"erro": f"Argumentos inválidos para {name}: {exc}"}
        except (KeyError, ValueError) as exc:
            return {"erro": f"Falha ao executar {name}: {exc}"}
