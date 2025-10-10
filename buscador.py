# buscador.py
from __future__ import annotations

from pathlib import Path
from typing import List, TypedDict
import os
import logging
import pandas as pd
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

mcp = FastMCP("buscador_livros")

DEFAULT_DATASET = Path(__file__).parent / "GoodReads_100k_books.csv"
DATASET_PATH = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))


class Book(TypedDict, total=False):
    title: str
    author: str
    pages: int
    genre: str
    rating: float
    desc: str


def _carregar_base() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, encoding="utf-8", on_bad_lines="skip")
    if "pages" in df.columns:
        df["pages"] = pd.to_numeric(df["pages"], errors="coerce").fillna(0).astype(int)
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


@mcp.tool()
def buscador(genero: str, numero_pg: int, topk: int = 10) -> List[Book]:
    """Busca por gênero (contains) e páginas ±20; aplica rating>=4.0 se houver >100 resultados; ordena por rating."""
    base = _carregar_base()
    gmask = base["genre"].astype(str).str.contains(genero, case=False, na=False)
    low, high = max(1, numero_pg - 20), numero_pg + 20
    pmask = base["pages"].between(low, high)
    out = base[gmask & pmask].copy()
    if len(out) > 100 and "rating" in out.columns:
        out = out[out["rating"] >= 4.0]
    out = (
        out.sort_values("rating", ascending=False)
        .loc[:, ["title", "author", "pages", "genre", "rating", "desc"]]
        .head(topk)
    )
    return out.to_dict(orient="records")


@mcp.tool()
def buscar_por_nome(titulo: str, topk: int = 10) -> List[Book]:
    """Busca por título (contains, case-insensitive) e retorna top-K por rating."""
    base = _carregar_base()
    mask = base["title"].astype(str).str.contains(titulo, case=False, na=False)
    out = (
        base[mask]
        .sort_values("rating", ascending=False)
        .loc[:, ["title", "author", "pages", "genre", "rating", "desc"]]
        .head(topk)
    )
    return out.to_dict(orient="records")


if __name__ == "__main__":
    mcp.run(transport="stdio")
