"""Loads the store's operational CSVs into typed, cleaned, in-memory records.

The dataset is tiny (~65 products, ~20 orders), so plain dataclasses + dicts are simpler, faster
and easier to test than a database or a dataframe. In production this module is the seam where
an ERP / database client would plug in: the tools only depend on the `StoreData` interface.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import DATA_DIR
from .text import digits, normalize

# Product.status values found in the data and how the agent should treat them.
SELLABLE_STATUS = "active"
STATUS_LABELS = {
    "active": "disponível no catálogo",
    "discontinued": "descontinuado (não faz mais parte do catálogo)",
    "coming_soon": "em breve (ainda não liberado para venda)",
}

# orders.status values -> customer-facing explanation (assumption documented in README).
ORDER_STATUS_LABELS = {
    "pending": "aguardando confirmação do pagamento",
    "confirmed": "pagamento confirmado, pedido em separação",
    "shipped": "enviado, a caminho",
    "delivered": "entregue",
    "cancelled": "cancelado",
}

PAYMENT_LABELS = {
    "pix": "PIX",
    "debit": "cartão de débito",
    "boleto": "boleto bancário",
}


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    description: str


@dataclass(frozen=True)
class Product:
    id: int
    name: str
    category_id: int
    category: str
    description: str
    price: float
    stock: int
    status: str
    specs: dict
    created_at: date

    @property
    def available(self) -> bool:
        return self.status == SELLABLE_STATUS and self.stock > 0

    @property
    def availability(self) -> str:
        if self.status != SELLABLE_STATUS:
            return STATUS_LABELS.get(self.status, self.status)
        if self.stock <= 0:
            return "esgotado no momento"
        if self.stock <= 2:
            return f"em estoque (últimas {self.stock} unidades)"
        return "em estoque"


@dataclass(frozen=True)
class Promotion:
    id: int
    product_id: int
    discount_percent: float
    name: str
    is_active: bool


@dataclass(frozen=True)
class Customer:
    id: int
    name: str
    phone: str
    email: str
    city: str

    @property
    def first_name(self) -> str:
        return self.name.split()[0]


@dataclass(frozen=True)
class OrderItem:
    product_id: int
    quantity: int


@dataclass
class Order:
    id: int
    customer_id: int
    order_date: date
    status: str
    total: float
    payment_method: str
    tracking_code: str | None
    estimated_delivery: date | None
    notes: str
    items: list[OrderItem] = field(default_factory=list)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh)]


def _date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def payment_label(method: str) -> str:
    if method.startswith("credit_"):
        return f"cartão de crédito em {method.split('_', 1)[1]}"
    return PAYMENT_LABELS.get(method, method)


class StoreData:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.categories: dict[int, Category] = {
            int(r["category_id"]): Category(int(r["category_id"]), r["name"], r["description"])
            for r in _read(data_dir / "categories.csv")
        }

        self.products: dict[int, Product] = {}
        for r in _read(data_dir / "products.csv"):
            try:
                specs = json.loads(r["specs"]) if r["specs"] else {}
            except json.JSONDecodeError:
                specs = {}
            cat_id = int(r["category_id"])
            self.products[int(r["product_id"])] = Product(
                id=int(r["product_id"]),
                name=r["name"],
                category_id=cat_id,
                category=self.categories[cat_id].name if cat_id in self.categories else "",
                description=r["description"],
                price=float(r["price_brl"]),
                stock=max(int(r["stock_quantity"] or 0), 0),
                status=r["status"].lower(),
                specs=specs,
                created_at=date.fromisoformat(r["created_at"]),
            )

        self.promotions: list[Promotion] = [
            Promotion(
                id=int(r["promotion_id"]),
                product_id=int(r["product_id"]),
                discount_percent=float(r["discount_percent"]),
                name=r["description"],
                is_active=r["is_active"] == "1",
            )
            for r in _read(data_dir / "promotions.csv")
        ]

        self.customers: dict[int, Customer] = {
            int(r["customer_id"]): Customer(
                int(r["customer_id"]), r["name"], r["phone"], r["email"].lower(), r["city"]
            )
            for r in _read(data_dir / "customers.csv")
        }

        self.orders: dict[int, Order] = {}
        for r in _read(data_dir / "orders.csv"):
            self.orders[int(r["order_id"])] = Order(
                id=int(r["order_id"]),
                customer_id=int(r["customer_id"]),
                order_date=date.fromisoformat(r["order_date"]),
                status=r["status"].lower(),
                total=float(r["total_brl"]),
                payment_method=r["payment_method"],
                tracking_code=r["tracking_code"] or None,
                estimated_delivery=_date(r["estimated_delivery"]),
                notes=r["notes"],
            )
        for r in _read(data_dir / "order_items.csv"):
            order = self.orders.get(int(r["order_id"]))
            if order:
                order.items.append(OrderItem(int(r["product_id"]), int(r["quantity"])))

    # ------------------------------------------------------------------ lookups

    def active_promotion(self, product_id: int) -> Promotion | None:
        """Promotions are not cumulative (policy 6.2): if several are active, the best one wins."""
        active = [p for p in self.promotions if p.product_id == product_id and p.is_active]
        return max(active, key=lambda p: p.discount_percent) if active else None

    def category_by_name(self, name: str) -> Category | None:
        wanted = normalize(name)
        for cat in self.categories.values():
            if normalize(cat.name) == wanted or wanted in normalize(cat.name):
                return cat
        return None

    def find_customers(self, email_or_phone: str) -> list[Customer]:
        """Identity check used before revealing any order data (LGPD, policy section 9).

        Returns a list because identifiers are not unique in the data (customers 11 and 43 share
        the same e-mail), so callers must never assume one identifier == one person.
        """
        value = (email_or_phone or "").strip().lower()
        if "@" in value:
            return [c for c in self.customers.values() if c.email == value]
        wanted = digits(value)
        if len(wanted) < 8:  # too short to identify someone safely
            return []
        # accept "(67) 99812-3456", "67998123456", "+55 67 99812 3456", "99812-3456"
        return [c for c in self.customers.values() if digits(c.phone).endswith(wanted[-11:]) or wanted.endswith(digits(c.phone))]
