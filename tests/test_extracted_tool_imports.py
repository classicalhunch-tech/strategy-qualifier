import importlib


MODULES = [
    "strategy_qualifier.strategies.fork_smc.signals",
    "strategy_qualifier.strategies.fork_smc.retest_engine",
    "strategy_qualifier.strategies.fork_smc.rules",
    "strategy_qualifier.strategies.fork_smc.confluence",
    "strategy_qualifier.strategies.fork_smc.mtf_structure",
    "strategy_qualifier.strategies.fork_smc.structure_targets",
    "strategy_qualifier.strategies.fork_smc.pattern_tags",
    "strategy_qualifier.research.ledger",
    "strategy_qualifier.research.block_bootstrap",
    "strategy_qualifier.research.deflated_sharpe",
    "strategy_qualifier.research.walk_forward",
    "strategy_qualifier.research.random_entry_baseline",
    "strategy_qualifier.research.rank_stability",
    "strategy_qualifier.research.monte_carlo",
    "strategy_qualifier.research.session_vs_outside",
    "strategy_qualifier.research.bif",
    "strategy_qualifier.research.exit_study",
    "strategy_qualifier.research.pattern_report",
    "strategy_qualifier.research.stop_ab_report",
    "strategy_qualifier.research.structure_target_report",
    "strategy_qualifier.research.compare_fill_modes",
    "strategy_qualifier.research.mtf_split",
    "strategy_qualifier.research.z_vs_room",
    "strategy_qualifier.research.z_vs_room_by_group",
]


def test_extracted_modules_import_without_original_repo():
    for name in MODULES:
        assert importlib.import_module(name) is not None
