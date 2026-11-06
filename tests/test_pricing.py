"""Tests for the LLM token pricing table (pathlayer.pricing)."""

from __future__ import annotations

import pytest

from pathlayer.pricing import (
    DEFAULT_PRICES,
    ModelPrice,
    cost_of,
    has_price,
    normalize_model_id,
    resolve_price,
)


def test_known_model_cost_is_input_plus_output() -> None:
    # gpt-4o: $2.5 / 1M input, $10 / 1M output.
    cost = cost_of("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(12.5)


def test_dated_alias_matches_by_longest_prefix() -> None:
    # A snapshot id resolves to its base model without listing every date.
    assert cost_of("gpt-4o-2024-08-06", 1_000_000, 0) == pytest.approx(2.5)


def test_prefix_match_prefers_longest_key() -> None:
    # "gpt-4o-mini" must win over "gpt-4o" for a mini snapshot.
    mini = cost_of("gpt-4o-mini-2024-07-18", 1_000_000, 0)
    assert mini == pytest.approx(DEFAULT_PRICES["gpt-4o-mini"].input_usd_per_mtok)


def test_case_insensitive() -> None:
    assert cost_of("GPT-4O", 1_000_000, 0) == pytest.approx(2.5)


def test_variant_does_not_borrow_base_model_price() -> None:
    # A different model whose id starts with a base key (non-dated suffix) must
    # NOT be priced as the base -- over-charging could trip a wrongful budget
    # STOP. o3-mini has its own (cheaper) entry, not o3's rate.
    assert cost_of("o3-mini", 1_000_000, 0) == pytest.approx(1.1)
    assert cost_of("o3", 1_000_000, 0) == pytest.approx(2.0)


def test_unknown_variant_of_known_base_is_free_not_overcharged() -> None:
    # o3-pro is not in the table; its non-digit suffix means it is NOT a dated
    # o3 snapshot, so it resolves to $0 (safe under-count) rather than o3's price.
    assert cost_of("o3-pro", 1_000_000, 1_000_000) == 0.0


def test_unknown_model_is_free() -> None:
    assert cost_of("myorg-secret-llm", 1_000_000, 1_000_000) == 0.0


def test_missing_model_is_free() -> None:
    assert cost_of(None, 1_000_000, 1_000_000) == 0.0
    assert cost_of("", 1_000_000, 1_000_000) == 0.0


def test_zero_or_negative_tokens_are_free() -> None:
    assert cost_of("gpt-4o", 0, 0) == 0.0
    # Defensive: a bogus negative count never produces a credit.
    assert cost_of("gpt-4o", -100, -100) == 0.0


def test_custom_prices_override_table() -> None:
    prices = {"house-model": ModelPrice(input_usd_per_mtok=1.0, output_usd_per_mtok=2.0)}
    assert cost_of("house-model", 1_000_000, 1_000_000, prices=prices) == pytest.approx(3.0)
    # The custom table replaces the default: a default model is now unknown.
    assert cost_of("gpt-4o", 1_000_000, 0, prices=prices) == 0.0


def test_custom_prices_keys_are_case_insensitive() -> None:
    # The case-insensitive promise must hold for caller-supplied keys too, not
    # just the model id -- an upper-case override key must still match.
    prices = {"GPT-4O": ModelPrice(input_usd_per_mtok=1.0, output_usd_per_mtok=2.0)}
    assert cost_of("gpt-4o", 1_000_000, 0, prices=prices) == pytest.approx(1.0)
    # And a dated snapshot still resolves against the upper-case override key.
    assert cost_of("gpt-4o-2024-08-06", 1_000_000, 0, prices=prices) == pytest.approx(1.0)


def test_claude_3_7_sonnet_is_priced() -> None:
    # Recent model added to the default table: $3 / 1M in, $15 / 1M out.
    assert cost_of("claude-3-7-sonnet", 1_000_000, 1_000_000) == pytest.approx(18.0)


def test_extra_prices_extends_without_dropping_defaults() -> None:
    # extra_prices layers on top of DEFAULT_PRICES: the new model is priced AND
    # known defaults keep working (unlike `prices=`, which replaces wholesale).
    extra = {"house-model": ModelPrice(input_usd_per_mtok=1.0, output_usd_per_mtok=2.0)}
    assert cost_of("house-model", 1_000_000, 1_000_000, extra_prices=extra) == pytest.approx(3.0)
    assert cost_of("gpt-4o", 1_000_000, 0, extra_prices=extra) == pytest.approx(2.5)


def test_extra_prices_overrides_a_default_entry() -> None:
    extra = {"gpt-4o": ModelPrice(input_usd_per_mtok=1.0, output_usd_per_mtok=1.0)}
    assert cost_of("gpt-4o", 1_000_000, 0, extra_prices=extra) == pytest.approx(1.0)


def test_extra_prices_pins_a_sub_version_the_table_does_not_list() -> None:
    # A family price drop is pinned explicitly so prefix-matching cannot inherit
    # the older, higher base rate and over-charge into a wrongful STOP. The
    # sub-version below is hypothetical: without a pin it borrows its base
    # claude-opus-5 ($5).
    extra = {"claude-opus-5-6": ModelPrice(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0)}
    assert cost_of("claude-opus-5-6-20270101", 1_000_000, 0) == pytest.approx(5.0)
    # With the pin, the correct (cheaper) rate wins via the longer key.
    assert cost_of("claude-opus-5-6-20270101", 1_000_000, 0, extra_prices=extra) == pytest.approx(
        3.0
    )


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate"),
    [
        # A sub-version cheaper than its base keeps its own rate: Opus 4.5 is
        # not Opus 4 ($15/$75), and Opus 5.5 is not Opus 5 ($5/$25).
        ("claude-opus-4-5-20251101", 5.0, 25.0),
        ("claude-opus-4-8", 5.0, 25.0),
        ("claude-opus-5-5", 4.0, 20.0),
        ("claude-opus-4-1-20250805", 15.0, 75.0),
        ("claude-opus-4-0", 15.0, 75.0),
        ("claude-sonnet-4-5-20250929", 3.0, 15.0),
        ("claude-sonnet-5-5", 2.0, 10.0),
        ("claude-haiku-4-5-20251001", 1.0, 5.0),
        ("claude-fable-5-1", 10.0, 50.0),
        ("claude-mythos-5", 10.0, 50.0),
        ("claude-mythos-5-1", 10.0, 50.0),
    ],
)
def test_claude_sub_versions_have_their_own_price(
    model: str, input_rate: float, output_rate: float
) -> None:
    assert cost_of(model, 1_000_000, 0) == pytest.approx(input_rate)
    assert cost_of(model, 0, 1_000_000) == pytest.approx(output_rate)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Amazon Bedrock: region profile, vendor segment, version qualifiers.
        ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5-20250929"),
        ("anthropic.claude-opus-5-5", "claude-opus-5-5"),
        ("us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5-20250929"),
        ("anthropic.claude-3-haiku-20240307-v1:0:200k", "claude-3-haiku-20240307"),
        # Routing prefixes, as OpenClaw's resolvedRef or OpenRouter write them.
        ("bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5-20250929"),
        ("openrouter/anthropic/claude-opus-4.5", "claude-opus-4-5"),
        # Google Vertex AI snapshot separator.
        ("claude-opus-4-5@20251101", "claude-opus-4-5"),
        # Nothing to unwrap: lowered, dots kept outside Claude ids, no -vN strip.
        ("GPT-4.1", "gpt-4.1"),
        ("deepseek-v3", "deepseek-v3"),
        ("eu.meta.llama3-70b-instruct-v1:0", "eu.meta.llama3-70b-instruct-v1:0"),
    ],
)
def test_normalize_model_id(model: str, expected: str) -> None:
    assert normalize_model_id(model) == expected


@pytest.mark.parametrize(
    ("model", "input_rate"),
    [
        # The id a live OpenClaw 2026.7.1 gateway on Bedrock reports, and its resolvedRef.
        ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0", 3.0),
        ("bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0", 3.0),
        ("us.anthropic.claude-3-5-haiku-20241022-v1:0", 0.8),
        ("global.anthropic.claude-opus-4-5-20251101-v1:0", 5.0),
        # Claude Mythos 5 and 5.1, raw and as Bedrock names them.
        ("anthropic.claude-mythos-5", 10.0),
        ("anthropic.claude-mythos-5-1", 10.0),
        ("us.anthropic.claude-mythos-5-1-v1:0", 10.0),
        ("bedrock/anthropic.claude-mythos-5", 10.0),
        ("anthropic/claude-sonnet-4", 3.0),
        ("anthropic/claude-sonnet-4.5", 3.0),
        # A raw snapshot match would read this as claude-opus-4 ($15).
        ("claude-opus-4-5@20251101", 5.0),
        ("claude-3-5-sonnet@20240620", 3.0),
    ],
)
def test_provider_wrapped_ids_resolve_to_the_vendor_price(model: str, input_rate: float) -> None:
    assert has_price(model)
    assert cost_of(model, 1_000_000, 0) == pytest.approx(input_rate)


@pytest.mark.parametrize(
    "model",
    [
        "eu.anthropic.claude-unknown-v1:0",
        "eu.meta.llama3-70b-instruct-v1:0",
        # The non-digit suffix rule survives unwrapping: o3-pro is not o3.
        "openai/o3-pro",
        "anthropic/claude-opus-4-mini",
        # A variant of a listed sub-version never falls back to its base:
        # Opus 4.5 to 4.8 would cost Opus 4's $15, Opus 5.5 Opus 5's $5.
        "claude-opus-4-5-thinking",
        "claude-opus-4.7-thinking",
        "openrouter/anthropic/claude-opus-4.5:nitro",
        "anthropic/claude-opus-4-6:floor",
        "claude-opus-5-5-thinking",
        "anthropic/claude-opus-5-5:floor",
        "claude-sonnet-4-5-20250929-thinking",
    ],
)
def test_unknown_wrapped_ids_stay_unpriced(model: str) -> None:
    assert not has_price(model)
    assert cost_of(model, 1_000_000, 1_000_000) == 0.0


def test_a_custom_vendor_key_prices_every_wrapper() -> None:
    prices = {"house-model": ModelPrice(1.0, 1.0)}
    for model in ("house-model-2026", "vertex/house-model@1", "eu.anthropic.house-model-v1:0"):
        assert cost_of(model, 1_000_000, 0, prices=prices) == pytest.approx(1.0)


def test_a_key_equal_to_the_raw_id_prices_that_id_alone() -> None:
    # A deployment can price one Bedrock profile (regional endpoints cost more)
    # without the rate leaking to other profiles or to first-party calls.
    regional = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    extra = {regional: ModelPrice(3.3, 16.5)}
    assert cost_of(regional, 1_000_000, 0, extra_prices=extra) == pytest.approx(3.3)
    for other in ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5-20250929"):
        assert cost_of(other, 1_000_000, 0, extra_prices=extra) == pytest.approx(3.0)


def test_extra_prices_layers_on_top_of_custom_prices() -> None:
    # When both are given, extra_prices extends the replacement table too.
    prices = {"house-model": ModelPrice(input_usd_per_mtok=1.0, output_usd_per_mtok=2.0)}
    extra = {"other-model": ModelPrice(input_usd_per_mtok=4.0, output_usd_per_mtok=0.0)}
    assert cost_of("other-model", 1_000_000, 0, prices=prices, extra_prices=extra) == pytest.approx(
        4.0
    )
    # gpt-4o is absent from both -> still unknown -> $0 (prices= replaced defaults).
    assert cost_of("gpt-4o", 1_000_000, 0, prices=prices, extra_prices=extra) == 0.0


def test_cache_tokens_are_priced_at_the_cache_rates() -> None:
    # Sonnet 4.5: $0.30 per 1M cache reads, $3.75 per 1M (5-minute) cache writes.
    assert cost_of("claude-sonnet-4-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.30)
    assert cost_of("claude-sonnet-4-5", 0, 0, cache_write_tokens=1_000_000) == pytest.approx(3.75)
    both = cost_of(
        "claude-sonnet-4-5",
        1_000_000,
        1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert both == pytest.approx(3.0 + 15.0 + 0.30 + 3.75)


def test_cache_tokens_without_a_cache_rate_are_left_out() -> None:
    # gpt-4o has no cache rate in the table: its cache tokens are unpriced, not
    # billed at the input rate.
    assert cost_of("gpt-4o", 0, 0, cache_read_tokens=1_000_000) == 0.0
    read_only = {"house": ModelPrice(1.0, 2.0, cache_read_usd_per_mtok=0.5)}
    cost = cost_of(
        "house", 0, 0, prices=read_only, cache_read_tokens=1_000_000, cache_write_tokens=1_000_000
    )
    assert cost == pytest.approx(0.5)


def test_negative_cache_tokens_are_free() -> None:
    assert cost_of("claude-sonnet-4-5", 0, 0, cache_read_tokens=-5, cache_write_tokens=-5) == 0.0


def test_resolve_price_returns_the_matched_entry() -> None:
    assert resolve_price("eu.anthropic.claude-opus-5-5-v1:0") == DEFAULT_PRICES["claude-opus-5-5"]
    assert resolve_price("house") is None
    custom = {"house": ModelPrice(1.0, 2.0)}
    assert resolve_price("house-2026", custom) == custom["house"]


@pytest.mark.parametrize("model", sorted(key for key in DEFAULT_PRICES if key.startswith("claude")))
def test_claude_cache_rates_follow_the_published_multipliers(model: str) -> None:
    # Guards against a typo in the table: the 5-minute write is 1.25x the input
    # rate, and a read 0.1x, except on the models with a cheaper read.
    price = DEFAULT_PRICES[model]
    read_multiplier = {
        "claude-fable-5-1": 0.025,
        "claude-mythos-5-1": 0.025,
        "claude-opus-5-5": 0.05,
        "claude-sonnet-5-5": 0.05,
    }
    assert price.cache_write_usd_per_mtok == pytest.approx(1.25 * price.input_usd_per_mtok)
    assert price.cache_read_usd_per_mtok == pytest.approx(
        read_multiplier.get(model, 0.1) * price.input_usd_per_mtok
    )
