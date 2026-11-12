"""Enterprise database generator for a 5-branch distributor (SQLite).

Creates a realistic, fully deterministic dataset:

* ``branches``     5 branches (A-E)
* ``products``     220 SKUs across 8 categories, with price and cost
* ``inventory``    one row per (branch, SKU): stock on hand + minimum alert level
* ``daily_sales``  ~30 days of line-level sales per branch
* ``dispatches``   outbound deliveries with status, route and driver

Seeding is reproducible (fixed RNG seed). Branch C is deliberately kept fully
stocked so the demo can show the agent answering "no records" honestly
instead of inventing alerts.

Usage::

    python seed_db.py                      # data/inventory.db, history ending today
    python seed_db.py --end-date 2026-11-12 --days 30 --db data/inventory.db
"""
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import config

SEED = 42
N_SKUS = 220

SCHEMA = """
CREATE TABLE branches (
    branch_id   TEXT PRIMARY KEY,          -- 'A'..'E'
    name        TEXT NOT NULL,
    city        TEXT NOT NULL
);

CREATE TABLE products (
    sku         TEXT PRIMARY KEY,          -- e.g. 'SKU-1001'
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    unit        TEXT NOT NULL,             -- each, box, roll, gallon...
    unit_price  REAL NOT NULL,             -- selling price (USD)
    unit_cost   REAL NOT NULL
);

CREATE TABLE inventory (
    branch_id         TEXT NOT NULL REFERENCES branches(branch_id),
    sku               TEXT NOT NULL REFERENCES products(sku),
    stock_on_hand     INTEGER NOT NULL,
    min_stock_alert   INTEGER NOT NULL,    -- reorder point
    last_restock_date TEXT NOT NULL,       -- ISO date
    PRIMARY KEY (branch_id, sku)
);

CREATE TABLE daily_sales (
    sale_id     INTEGER PRIMARY KEY,
    sale_date   TEXT NOT NULL,             -- ISO date
    branch_id   TEXT NOT NULL REFERENCES branches(branch_id),
    sku         TEXT NOT NULL REFERENCES products(sku),
    qty         INTEGER NOT NULL,
    unit_price  REAL NOT NULL,
    revenue     REAL NOT NULL              -- qty * unit_price
);

CREATE TABLE dispatches (
    dispatch_id   TEXT PRIMARY KEY,        -- e.g. 'DSP-A-00012'
    branch_id     TEXT NOT NULL REFERENCES branches(branch_id),
    dispatch_date TEXT NOT NULL,           -- ISO date
    customer      TEXT NOT NULL,
    route         TEXT NOT NULL,           -- e.g. 'R-A2 North'
    driver        TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN
                  ('PENDING','IN_TRANSIT','DELIVERED','DELAYED','CANCELLED')),
    items_count   INTEGER NOT NULL,
    order_value   REAL NOT NULL
);

CREATE INDEX ix_sales_date_branch ON daily_sales(sale_date, branch_id);
CREATE INDEX ix_dispatch_date_branch ON dispatches(dispatch_date, branch_id);
"""

BRANCHES = [
    ("A", "Branch A - Central Warehouse", "Monterrey"),
    ("B", "Branch B - North Hub", "Saltillo"),
    ("C", "Branch C - Port Depot", "Veracruz"),
    ("D", "Branch D - Bajio Center", "Queretaro"),
    ("E", "Branch E - Pacific Outlet", "Guadalajara"),
]

CATEGORIES = {
    "Fasteners": (["Hex Bolt", "Lock Nut", "Flat Washer", "Wood Screw", "Anchor", "Rivet"],
                  ["M6", "M8", "M10", "M12", '1/4"', '3/8"'], "box", (4, 40)),
    "Electrical": (["THHN Wire", "Breaker", "Conduit", "Junction Box", "LED Panel", "Switch"],
                   ["12AWG", "20A", '3/4"', "4x4", "60W", "3-Way"], "each", (3, 180)),
    "Plumbing": (["PVC Pipe", "Ball Valve", "Elbow", "Coupling", "P-Trap", "Teflon Tape"],
                 ['1/2"', '3/4"', '1"', '2"', "90deg", "Brass"], "each", (1, 60)),
    "Packaging": (["Stretch Film", "Carton Box", "Packing Tape", "Bubble Wrap", "Pallet Strap"],
                  ["18in", "Large", "48mm", "12in", "Poly"], "roll", (2, 45)),
    "Safety": (["Nitrile Gloves", "Safety Glasses", "Hard Hat", "Hi-Vis Vest", "Ear Plugs", "Respirator"],
               ["L", "Clear", "Type I", "Class 2", "NRR32", "N95"], "box", (3, 70)),
    "Tools": (["Drill Bit Set", "Tape Measure", "Utility Knife", "Torque Wrench", "Level", "Pliers"],
              ["HSS", "25ft", "Pro", '1/2"', "24in", "8in"], "each", (6, 220)),
    "Lubricants": (["Hydraulic Oil", "Grease Cartridge", "Gear Oil", "Penetrating Oil", "Chain Lube"],
                   ["ISO 46", "EP2", "80W-90", "12oz", "Spray"], "gallon", (5, 90)),
    "Cleaning": (["Degreaser", "Shop Towels", "Floor Cleaner", "Hand Soap", "Absorbent Pads"],
                 ["Citrus", "Blue", "Neutral", "Gallon", "Universal"], "each", (2, 40)),
}

DRIVERS = ["Luis Ortega", "Ana Cruz", "Miguel Soto", "Carla Ruiz", "Jorge Pena", "Diana Leal",
           "Raul Vega", "Sofia Mora", "Pedro Rios", "Elena Diaz"]
CUSTOMERS = ["Ferreteria El Toro", "Constructora Delta", "Industrias Norte", "Taller Hermanos Gil",
             "Grupo Alfa Mantenimiento", "Planta Automotriz Sur", "Hotel Costa Azul",
             "Mercado Central", "Agroindustrias Lopez", "Logistica Rapida"]
ROUTE_ZONES = ["North", "South", "East", "West", "Downtown", "Industrial Park"]


def build_products(rng: random.Random) -> list[tuple]:
    products = []
    cats = list(CATEGORIES.items())
    for i in range(N_SKUS):
        cat, (nouns, variants, unit, (lo, hi)) = cats[i % len(cats)]
        noun = nouns[(i // len(cats)) % len(nouns)]
        variant = variants[(i // (len(cats) * len(nouns))) % len(variants)]
        grade = ["Std", "Pro", "Heavy Duty", "Eco"][(i // 7) % 4]
        price = round(rng.uniform(lo, hi), 2)
        cost = round(price * rng.uniform(0.55, 0.75), 2)
        products.append((f"SKU-{1001 + i}", f"{noun} {variant} {grade}", cat, unit, price, cost))
    return products


def build_inventory(rng: random.Random, products: list[tuple], end: date) -> list[tuple]:
    rows = []
    for branch_id, *_ in BRANCHES:
        for sku, *_rest in products:
            min_alert = rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 80, 100])
            roll = rng.random()
            if branch_id == "C" or roll > 0.12:          # healthy stock
                stock = rng.randint(min_alert + 1, min_alert * 5)
            elif roll > 0.04:                              # low: at/below reorder point
                stock = rng.randint(max(1, min_alert // 2), min_alert)
            else:                                          # critical: < 50 % of reorder point
                stock = rng.randint(0, max(0, min_alert // 2 - 1))
            restock = end - timedelta(days=rng.randint(1, 45))
            rows.append((branch_id, sku, stock, min_alert, restock.isoformat()))
    return rows


def build_sales(rng: random.Random, products: list[tuple], end: date, days: int) -> list[tuple]:
    rows = []
    branch_volume = {"A": 1.6, "B": 1.1, "C": 0.8, "D": 1.3, "E": 0.9}
    for d in range(days - 1, -1, -1):
        day = end - timedelta(days=d)
        weekend = day.weekday() >= 5
        for branch_id, *_ in BRANCHES:
            n_lines = int(rng.randint(18, 34) * branch_volume[branch_id] * (0.45 if weekend else 1))
            for _ in range(n_lines):
                sku, _n, _c, _u, price, _cost = rng.choice(products)
                qty = rng.randint(1, 24)
                rows.append((day.isoformat(), branch_id, sku, qty, price, round(qty * price, 2)))
    return rows


def build_dispatches(rng: random.Random, end: date, days: int) -> list[tuple]:
    rows = []
    seq = {b: 0 for b, *_ in BRANCHES}
    for d in range(days - 1, -1, -1):
        day = end - timedelta(days=d)
        for branch_id, *_ in BRANCHES:
            for k in range(rng.randint(3, 8)):
                seq[branch_id] += 1
                if d == 0:      # today: still moving
                    status = rng.choices(["PENDING", "IN_TRANSIT", "DELIVERED", "DELAYED"],
                                         [3, 4, 2, 1])[0]
                    if k == 0 and branch_id in ("A", "D"):   # guarantee a few exceptions for the demo
                        status = "DELAYED"
                elif d == 1:
                    status = rng.choices(["IN_TRANSIT", "DELIVERED", "DELAYED"], [2, 7, 1])[0]
                else:
                    status = rng.choices(["DELIVERED", "CANCELLED", "DELAYED"], [94, 4, 2])[0]
                zone = rng.choice(ROUTE_ZONES)
                rows.append((
                    f"DSP-{branch_id}-{seq[branch_id]:05d}", branch_id, day.isoformat(),
                    rng.choice(CUSTOMERS), f"R-{branch_id}{ROUTE_ZONES.index(zone) + 1} {zone}",
                    rng.choice(DRIVERS), status, rng.randint(2, 40), round(rng.uniform(150, 9500), 2),
                ))
    return rows


def seed(db_path: Path = config.DB_PATH, end_date: date | None = None, days: int = 30) -> Path:
    """(Re)create the SQLite database and return its path."""
    end = end_date or date.today()
    rng = random.Random(SEED)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    products = build_products(rng)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO branches VALUES (?,?,?)", BRANCHES)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)
        conn.executemany("INSERT INTO inventory VALUES (?,?,?,?,?)", build_inventory(rng, products, end))
        conn.executemany(
            "INSERT INTO daily_sales (sale_date, branch_id, sku, qty, unit_price, revenue) VALUES (?,?,?,?,?,?)",
            build_sales(rng, products, end, days),
        )
        conn.executemany("INSERT INTO dispatches VALUES (?,?,?,?,?,?,?,?,?)", build_dispatches(rng, end, days))
        conn.commit()
    finally:
        conn.close()
    return db_path


def _summary(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("branches", "products", "inventory", "daily_sales", "dispatches")}
        low = conn.execute("SELECT COUNT(*) FROM inventory WHERE stock_on_hand <= min_stock_alert").fetchone()[0]
        rng_dates = conn.execute("SELECT MIN(sale_date), MAX(sale_date) FROM daily_sales").fetchone()
    finally:
        conn.close()
    lines = [f"  {t:<12} {n:>6} rows" for t, n in counts.items()]
    lines.append(f"  low-stock rows: {low}  |  sales range: {rng_dates[0]} -> {rng_dates[1]}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    ap.add_argument("--end-date", type=date.fromisoformat, default=None, help="last day of history (default: today)")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    path = seed(args.db, args.end_date, args.days)
    print(f"Seeded {path}\n{_summary(path)}")


if __name__ == "__main__":
    main()
