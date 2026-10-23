import pytest

from backend.app.data.instruments.isin_links import (
    ISIN_SUCCESSOR_LINKS,
    all_linked_isins,
    expand_isin_chain,
)


def test_isin_with_no_predecessor_returns_itself():
    assert expand_isin_chain("INE002A01018") == {"INE002A01018"}


def test_bajfinance_chain_includes_all_three_isins():
    chain = expand_isin_chain("INE296A01032")
    assert chain == {"INE296A01032", "INE296A01024", "INE296A01016"}


def test_hdfcbank_chain_includes_old_and_new_isin():
    chain = expand_isin_chain("INE040A01034")
    assert chain == {"INE040A01034", "INE040A01026"}


def test_kotakbank_chain_confirmed_complete_by_exhaustive_scan():
    chain = expand_isin_chain("INE237A01036")
    assert chain == {"INE237A01036", "INE237A01028"}


def test_jswsteel_chain_includes_old_and_new_isin():
    chain = expand_isin_chain("INE019A01038")
    assert chain == {"INE019A01038", "INE019A01020"}


def test_every_curated_link_resolves_without_error():
    for new_isin in ISIN_SUCCESSOR_LINKS:
        chain = expand_isin_chain(new_isin)
        assert new_isin in chain
        assert len(chain) >= 2


def test_cycle_raises():
    ISIN_SUCCESSOR_LINKS["A"] = ("B", "test cycle")
    ISIN_SUCCESSOR_LINKS["B"] = ("A", "test cycle")
    try:
        with pytest.raises(ValueError, match="cycle"):
            expand_isin_chain("A")
    finally:
        del ISIN_SUCCESSOR_LINKS["A"]
        del ISIN_SUCCESSOR_LINKS["B"]


class TestAllLinkedIsins:

    def test_starting_from_oldest_isin_finds_the_full_chain(self):
        assert all_linked_isins("INE296A01016") == {
            "INE296A01016",
            "INE296A01024",
            "INE296A01032",
        }

    def test_starting_from_middle_isin_finds_the_full_chain(self):
        assert all_linked_isins("INE296A01024") == {
            "INE296A01016",
            "INE296A01024",
            "INE296A01032",
        }

    def test_starting_from_newest_isin_finds_the_full_chain(self):
        assert all_linked_isins("INE296A01032") == {
            "INE296A01016",
            "INE296A01024",
            "INE296A01032",
        }

    def test_unlinked_isin_returns_itself_only(self):
        assert all_linked_isins("INE002A01018") == {"INE002A01018"}

    def test_two_isin_chain_from_either_end(self):
        assert all_linked_isins("INE040A01026") == {"INE040A01026", "INE040A01034"}
        assert all_linked_isins("INE040A01034") == {"INE040A01026", "INE040A01034"}
