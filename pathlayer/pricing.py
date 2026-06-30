"""LLM token pricing (local-first, zero network).

Turns token counts into a USD cost so an adapter can populate
:attr:`~pathlayer.events.ToolEvent.cost_usd` for LLM calls and
``BudgetDetector`` can account for model spend, not just tool spend.

Prices are a static, versioned table expressed in **USD per 1M tokens**, with
separate input (prompt) and output (completion) rates. The table is deliberately
small, **non-exhaustive**, and overridable two ways via ``cost_of``:

- ``prices=`` **replaces** ``DEFAULT_PRICES`` entirely (full custom table).
- ``extra_prices=`` **extends** the base, adding/overriding individual models
  while keeping the known defaults -- the ergonomic path for "I just need one
  more model" (e.g. a 2026 model not yet listed, or a house model).

There is no network lookup -- a stale price is preferable to a runtime dependency
on a pricing API in the detection path. The table is updated by hand; newer
models (2026+) are expected to be supplied by the caller via ``extra_prices``
rather than being guessed here.

Watch out for *family price drops*: prefix matching resolves a dated/sub-version
id to its base key (``claude-opus-4-5-...`` -> ``claude-opus-4``). If a newer
sub-version is cheaper than its base, that inherits the **older, higher** price
and over-charges -- which could trip a wrongful budget STOP. Pin the correct
price explicitly via ``extra_prices`` (e.g. ``{"claude-opus-4-5": ModelPrice(
5.0, 25.0)}``) rather than relying on the inherited base rate.

Unknown model or missing token usage resolves to ``0.0`` (documented, never a
crash): we would rather under-count spend than guess. The trade-off is that a
provider which does not report a usable model id leaves ``BudgetDetector`` blind
for that call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per 1M tokens, split by input (prompt) and output (completion)."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float


# Static, versioned price table (USD per 1M tokens). Common OpenAI / Anthropic
# models as of 2025. Keys are matched case-insensitively and by longest known
# prefix (see ``_lookup``) so dated aliases like ``gpt-4o-2024-08-06`` resolve to
# ``gpt-4o``. Override or extend via the ``prices`` argument of ``cost_of``.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(2.5, 10.0),
    "gpt-4o-mini": ModelPrice(0.15, 0.6),
    "gpt-4.1": ModelPrice(2.0, 8.0),
    "gpt-4.1-mini": ModelPrice(0.4, 1.6),
    "gpt-4.1-nano": ModelPrice(0.1, 0.4),
    "o3": ModelPrice(2.0, 8.0),
    "o3-mini": ModelPrice(1.1, 4.4),
    "o4-mini": ModelPrice(1.1, 4.4),
    "claude-3-7-sonnet": ModelPrice(3.0, 15.0),
    "claude-3-5-sonnet": ModelPrice(3.0, 15.0),
    "claude-3-5-haiku": ModelPrice(0.8, 4.0),
    "claude-3-opus": ModelPrice(15.0, 75.0),
    "claude-3-haiku": ModelPrice(0.25, 1.25),
    "claude-sonnet-4": ModelPrice(3.0, 15.0),
    "claude-opus-4": ModelPrice(15.0, 75.0),
}

_PER_MTOK = 1_000_000.0


def _lookup(model: str | None, prices: Mapping[str, ModelPrice]) -> ModelPrice | None:
    """Resolve ``model`` to a price, case-insensitively.

    Exact match first. Otherwise treat the id as a *dated snapshot* of a table
    key only when the suffix after ``"<key>-"`` starts with a digit, e.g.
    ``gpt-4o-2024-08-06`` -> ``gpt-4o``. A non-digit suffix marks a *different*
    model (``o3-mini`` is not ``o3``, ``gpt-4o-mini`` is not ``gpt-4o``), so it
    must NOT borrow the base price -- mispricing toward over-charge could trip
    ``BudgetDetector`` into a wrongful STOP. Unknown ids return ``None`` (the
    caller bills $0, an under-count we accept over a wrong stop). Longest key
    wins on ties. Returns ``None`` when nothing matches.

    Both the ``model`` id and the table keys are lowered before comparison, so a
    custom ``prices`` override with upper-case keys (``{"GPT-4O": ...}``) is
    matched too -- otherwise the case-insensitive promise would silently hold
    for the model id but not for caller-supplied keys, mispricing to ``$0``.
    """
    if not model:
        return None
    lowered = model.lower()
    table = {key.lower(): value for key, value in prices.items()}
    exact = table.get(lowered)
    if exact is not None:
        return exact
    best_key = ""
    for key in table:
        prefix = f"{key}-"
        suffix = lowered[len(prefix) :]
        if lowered.startswith(prefix) and suffix[:1].isdigit() and len(key) > len(best_key):
            best_key = key
    return table[best_key] if best_key else None


def cost_of(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    prices: Mapping[str, ModelPrice] | None = None,
    extra_prices: Mapping[str, ModelPrice] | None = None,
) -> float:
    """USD cost of a single LLM call, or ``0.0`` if it cannot be priced.

    Returns ``0.0`` for an unknown/absent model or non-positive token counts,
    rather than raising: a missing price must never crash the capture path.

    Price resolution:

    - ``prices`` given: that mapping **replaces** ``DEFAULT_PRICES`` (full table).
    - ``extra_prices`` given: those entries are layered **on top of** the base
      (``prices`` if also given, else ``DEFAULT_PRICES``), adding or overriding
      individual models without dropping the rest. Use it to price a model the
      default table does not list yet.
    """
    base = prices if prices is not None else DEFAULT_PRICES
    table = {**base, **extra_prices} if extra_prices else base
    price = _lookup(model, table)
    if price is None:
        return 0.0
    billable_in = max(0, input_tokens)
    billable_out = max(0, output_tokens)
    return (
        billable_in * price.input_usd_per_mtok + billable_out * price.output_usd_per_mtok
    ) / _PER_MTOK
