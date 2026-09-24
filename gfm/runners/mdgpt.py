"""gfm entry point for the paper-based MDGPT reproduction."""

from gfm.vendor.mdgpt import GCN, MDGPT, DualPrompt, run_mdgpt

__all__ = ["MDGPT", "GCN", "DualPrompt", "run_mdgpt"]
