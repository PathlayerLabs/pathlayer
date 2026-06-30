"""Tests for the LLM token pricing table (pathlayer.pricing)."""

from __future__ import annotations

import pytest

from pathlayer.pricing import DEFAULT_PRICES, ModelPrice, cost_of


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
    # A family price drop is pinned explicitly so prefix-matching cannot inherit
    # the older, higher base rate and over-charge into a wrongful STOP.
    extra = {"claude-opus-4-5": ModelPrice(input_usd_per_mtok=5.0, output_usd_per_mtok=25.0)}
    # Without the override, a dated opus-4.5 id prefix-matches claude-opus-4 ($15/$75).
    assert cost_of("claude-opus-4-5-20260101", 1_000_000, 0) == pytest.approx(15.0)
    # With it, the correct (cheaper) rate wins via the longer exact-prefix key.
    assert cost_of("claude-opus-4-5-20260101", 1_000_000, 0, extra_prices=extra) == pytest.approx(
        5.0
    )


def test_extra_prices_layers_on_top_of_custom_prices() -> None:
    # When both are given, extra_prices extends the replacement table too.
    prices = {"house-model": ModelPrice(input_usd_per_mtok=1.0, output_usd_per_mtok=2.0)}
    extra = {"other-model": ModelPrice(input_usd_per_mtok=4.0, output_usd_per_mtok=0.0)}
    assert cost_of("other-model", 1_000_000, 0, prices=prices, extra_prices=extra) == pytest.approx(
        4.0
    )
    # gpt-4o is absent from both -> still unknown -> $0 (prices= replaced defaults).
    assert cost_of("gpt-4o", 1_000_000, 0, prices=prices, extra_prices=extra) == 0.0
