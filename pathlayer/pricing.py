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
on a pricing API in the detection path. The table is updated by hand; a model it
does not list yet is expected to be supplied by the caller via ``extra_prices``
rather than being guessed here.

Watch out for *family price drops*: prefix matching resolves a dated/sub-version
id to its base key (``claude-opus-5-6-...`` -> ``claude-opus-5``). If a newer
sub-version is cheaper than its base, that inherits the **older, higher** price
and over-charges -- which could trip a wrongful budget STOP. That is why every
Claude sub-version is listed explicitly (Opus 4.5 is $5 / $25, not Opus 4's
$15 / $75). For a model the table does not list, pin the price via
``extra_prices`` rather than relying on an inherited base rate.

Cloud providers wrap the vendor's own id: Amazon Bedrock
(``eu.anthropic.claude-sonnet-4-5-20250929-v1:0``), Google Vertex AI
(``claude-opus-4-5@20251101``), routing prefixes (``bedrock/...``,
``anthropic/...``) and dotted versions (``anthropic/claude-sonnet-4.5``). They
resolve to the vendor's id (see :func:`normalize_model_id`) and are priced at
the vendor's first-party list rates. A partner platform bills separately
(Bedrock regional endpoints add about 10 percent), so that rate errs low. The
estimate as a whole also depends on the token counts an adapter passes: billing
prompt-cache tokens at the input rate over-counts whatever the rate.

Unknown model or missing token usage resolves to ``0.0`` (documented, never a
crash): we would rather under-count spend than guess. The trade-off is that a
provider which does not report a usable model id leaves ``BudgetDetector`` blind
for that call.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per 1M tokens: input (prompt), output (completion) and prompt cache.

    ``cache_read_usd_per_mtok`` prices tokens served from the prompt cache and
    ``cache_write_usd_per_mtok`` tokens written to it. ``None`` means the rate is
    unknown, not free: :func:`cost_of` then leaves those tokens unpriced, an
    under-count rather than a guess. Billing them at the input rate instead
    would over-charge cache reads up to tenfold.
    """

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_usd_per_mtok: float | None = None
    cache_write_usd_per_mtok: float | None = None


# Static, versioned price table (USD per 1M tokens). Keys are matched
# case-insensitively and by longest known prefix (see ``_lookup``) so dated
# aliases like ``gpt-4o-2024-08-06`` resolve to ``gpt-4o``. Override or extend via
# the ``prices`` argument of ``cost_of``. Anthropic rates checked against
# https://platform.claude.com/docs/en/about-claude/pricing on 2026-11-04,
# including its cache rates: a cache read costs 0.1x the input rate (0.05x on
# Opus 5.5 and Sonnet 5.5, 0.025x on Fable 5.1 and Mythos 5.1) and the cache
# write is the 5-minute one, 1.25x (the 1-hour write costs 2x, so an unknown TTL
# errs low).
# OpenAI entries carry no cache rate yet. Claude Haiku 5.5 is left out: its rate
# depends on the prompt length, which a flat per-token entry cannot express.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(2.5, 10.0),
    "gpt-4o-mini": ModelPrice(0.15, 0.6),
    "gpt-4.1": ModelPrice(2.0, 8.0),
    "gpt-4.1-mini": ModelPrice(0.4, 1.6),
    "gpt-4.1-nano": ModelPrice(0.1, 0.4),
    "o3": ModelPrice(2.0, 8.0),
    "o3-mini": ModelPrice(1.1, 4.4),
    "o4-mini": ModelPrice(1.1, 4.4),
    "claude-3-7-sonnet": ModelPrice(3.0, 15.0, 0.30, 3.75),
    "claude-3-5-sonnet": ModelPrice(3.0, 15.0, 0.30, 3.75),
    "claude-3-5-haiku": ModelPrice(0.8, 4.0, 0.08, 1.0),
    "claude-3-opus": ModelPrice(15.0, 75.0, 1.50, 18.75),
    "claude-3-haiku": ModelPrice(0.25, 1.25, 0.025, 0.3125),
    "claude-sonnet-4": ModelPrice(3.0, 15.0, 0.30, 3.75),
    "claude-sonnet-4-5": ModelPrice(3.0, 15.0, 0.30, 3.75),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 0.30, 3.75),
    "claude-sonnet-5": ModelPrice(2.0, 10.0, 0.20, 2.50),
    "claude-sonnet-5-5": ModelPrice(2.0, 10.0, 0.10, 2.50),
    "claude-opus-4": ModelPrice(15.0, 75.0, 1.50, 18.75),
    "claude-opus-4-1": ModelPrice(15.0, 75.0, 1.50, 18.75),
    "claude-opus-4-5": ModelPrice(5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-6": ModelPrice(5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-7": ModelPrice(5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-8": ModelPrice(5.0, 25.0, 0.50, 6.25),
    "claude-opus-5": ModelPrice(5.0, 25.0, 0.50, 6.25),
    "claude-opus-5-5": ModelPrice(4.0, 20.0, 0.20, 5.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 0.10, 1.25),
    "claude-fable-5": ModelPrice(10.0, 50.0, 1.0, 12.50),
    "claude-fable-5-1": ModelPrice(10.0, 50.0, 0.25, 12.50),
    # Limited availability: verified organizations only.
    "claude-mythos-5": ModelPrice(10.0, 50.0, 1.0, 12.50),
    "claude-mythos-5-1": ModelPrice(10.0, 50.0, 0.25, 12.50),
}

_PER_MTOK = 1_000_000.0

# Amazon Bedrock ids: an optional cross-region inference profile (``eu.``,
# ``us.``, ``apac.``, ``global.``, ``us-gov.``...) then the vendor segment.
_BEDROCK_ANTHROPIC = re.compile(r"^(?:[a-z]{2,6}(?:-[a-z]{2,6})?\.)?anthropic\.(?P<model>.+)$")
# Bedrock's version suffix once the ``:...`` qualifiers are gone (``-v1``).
_BEDROCK_VERSION = re.compile(r"-v\d+$")


def normalize_model_id(model: str) -> str:
    """The vendor's own id for a provider-wrapped model id, lowered.

    Drops a routing prefix up to the last ``/`` (``bedrock/``, ``anthropic/``),
    a Vertex AI snapshot (``@20251101``), and a Bedrock wrapper (region profile,
    ``anthropic.`` vendor segment, ``-v1:0`` version). A Claude id with a dotted
    version (``claude-sonnet-4.5``) gets dashes, its first-party spelling; other
    ids keep their dots (``gpt-4.1``). An id with no wrapper comes back lowered
    and otherwise unchanged.
    """
    normalized = model.lower().rsplit("/", 1)[-1].split("@", 1)[0]
    bedrock = _BEDROCK_ANTHROPIC.match(normalized)
    if bedrock is not None:
        normalized = _BEDROCK_VERSION.sub("", bedrock.group("model").split(":", 1)[0])
    if normalized.startswith("claude-"):
        normalized = normalized.replace(".", "-")
    return normalized


# What may follow ``<key>-`` in a dated snapshot of that key: digits and dashes
# only (``20251101``, ``2024-08-06``, ``1-20250805``).
_SNAPSHOT_SUFFIX = re.compile(r"\d[\d-]*")


def _match(
    model: str, table: Mapping[str, ModelPrice], *, legacy: bool = False
) -> ModelPrice | None:
    """Exact key, else the longest key ``model`` is a dated snapshot of.

    ``legacy`` applies Pathlayer 0.1.0's looser rule: any suffix that starts
    with a digit, variants of a listed sub-version included.
    """
    exact = table.get(model)
    if exact is not None:
        return exact
    best_key = ""
    for key in table:
        prefix = f"{key}-"
        suffix = model[len(prefix) :]
        dated = suffix[:1].isdigit() if legacy else _SNAPSHOT_SUFFIX.fullmatch(suffix)
        if model.startswith(prefix) and dated and len(key) > len(best_key):
            best_key = key
    return table[best_key] if best_key else None


def _lookup(
    model: str | None, prices: Mapping[str, ModelPrice], *, unwrap_ids: bool = True
) -> ModelPrice | None:
    """Resolve ``model`` to a price, case-insensitively.

    Exact match first. Otherwise treat the id as a *dated snapshot* of a table
    key only when the suffix after ``"<key>-"`` is made of digits and dashes,
    e.g. ``gpt-4o-2024-08-06`` -> ``gpt-4o``. Anything else marks a *different*
    model (``o3-mini`` is not ``o3``, ``gpt-4o-mini`` is not ``gpt-4o``), so it
    must NOT borrow the base price -- mispricing toward over-charge could trip
    ``BudgetDetector`` into a wrongful STOP. That includes a variant of a listed
    sub-version (``claude-opus-4-5-thinking``, OpenRouter's
    ``claude-opus-4.5:nitro``): read as ``5-thinking`` after
    ``claude-opus-4-``, it would borrow Opus 4's price, three times Opus 4.5's.
    Unknown ids return ``None`` (the caller bills $0, an under-count we accept
    over a wrong stop). Longest key wins on ties. Returns ``None`` when nothing
    matches.

    Both the ``model`` id and the table keys are lowered before comparison, so a
    custom ``prices`` override with upper-case keys (``{"GPT-4O": ...}``) is
    matched too -- otherwise the case-insensitive promise would silently hold
    for the model id but not for caller-supplied keys, mispricing to ``$0``.

    A key equal to the raw id wins first, so a key that names a provider-wrapped
    id exactly (one Bedrock profile's rate, say) applies to that id alone. Then
    the id is normalized (:func:`normalize_model_id`) and matched as above
    against the vendor ids of the table, so ``claude-sonnet-4-5`` prices every
    wrapper of that model. The keys themselves are not normalized: a rate
    written for ``eu.anthropic....`` must not leak to other profiles or to
    first-party calls. The raw id never goes through the snapshot rule either:
    ``claude-opus-4-5@20251101`` would read as a snapshot of ``claude-opus-4``
    and borrow its higher price.

    ``unwrap_ids=False`` matches as Pathlayer 0.1.0 did: no normalization, and
    its looser snapshot rule on the raw id. A report rebuilt for a run recorded
    then uses it, so its figures are the ones that recording computed: no call
    it left unpriced gains a price, none it priced loses one.
    """
    if not model:
        return None
    lowered = model.lower()
    table = {key.lower(): value for key, value in prices.items()}
    if not unwrap_ids:
        return _match(lowered, table, legacy=True)
    exact = table.get(lowered)
    if exact is not None:
        return exact
    return _match(normalize_model_id(lowered), table)


def resolve_price(
    model: str | None,
    prices: Mapping[str, ModelPrice] | None = None,
    *,
    unwrap_ids: bool = True,
) -> ModelPrice | None:
    """The price ``model`` resolves to, or ``None`` when it has none.

    ``prices`` replaces ``DEFAULT_PRICES``, as in :func:`cost_of`. Adapters read
    the cache rates here to say which tokens a cost leaves unpriced.
    """
    return _lookup(model, prices if prices is not None else DEFAULT_PRICES, unwrap_ids=unwrap_ids)


def has_price(model: str | None, prices: Mapping[str, ModelPrice] | None = None) -> bool:
    """Whether ``model`` resolves to a configured price.

    Adapters use this to distinguish an explicit free model from an unknown
    model whose zero value must remain *unmeasured*, not be presented as free.
    """
    return resolve_price(model, prices) is not None


def cost_of(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    prices: Mapping[str, ModelPrice] | None = None,
    extra_prices: Mapping[str, ModelPrice] | None = None,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    unwrap_ids: bool = True,
) -> float:
    """USD cost of a single LLM call, or ``0.0`` if it cannot be priced.

    Returns ``0.0`` for an unknown/absent model or non-positive token counts,
    rather than raising: a missing price must never crash the capture path.
    Prompt-cache tokens are priced at the model's cache rates; a model whose
    cache rate is unknown (``None``) leaves those tokens out of the cost.
    ``unwrap_ids`` is passed to the lookup (see ``_lookup``).

    Price resolution:

    - ``prices`` given: that mapping **replaces** ``DEFAULT_PRICES`` (full table).
    - ``extra_prices`` given: those entries are layered **on top of** the base
      (``prices`` if also given, else ``DEFAULT_PRICES``), adding or overriding
      individual models without dropping the rest. Use it to price a model the
      default table does not list yet.
    """
    base = prices if prices is not None else DEFAULT_PRICES
    table = {**base, **extra_prices} if extra_prices else base
    price = _lookup(model, table, unwrap_ids=unwrap_ids)
    if price is None:
        return 0.0
    usd_per_mtok = max(0, input_tokens) * price.input_usd_per_mtok
    usd_per_mtok += max(0, output_tokens) * price.output_usd_per_mtok
    if price.cache_read_usd_per_mtok is not None:
        usd_per_mtok += max(0, cache_read_tokens) * price.cache_read_usd_per_mtok
    if price.cache_write_usd_per_mtok is not None:
        usd_per_mtok += max(0, cache_write_tokens) * price.cache_write_usd_per_mtok
    return usd_per_mtok / _PER_MTOK
