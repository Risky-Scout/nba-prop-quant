"""Research-only subpackages.

Nothing under ``nba_prop_quant.research`` is imported by the production
serving path, the Step 3C adaptive trainer or the Step 3D automation. These
modules exist so shadow studies can reuse certified production code without
editing it.
"""
