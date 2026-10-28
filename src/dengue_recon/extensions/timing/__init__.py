"""Timing-informed share model (EXPLORATORY; developed after the pre-registered analysis).

Reconstructs a municipality's weekly dengue curve from its known annual total
with the penalised Dirichlet-multinomial share model (B4s structure) plus
same-year timing information from a higher level: the weekly curve of the other
municipalities of the same state (S1), the state's Google Trends search curve
(S2), both (S3), both plus climate (S3C), or a year-specific shift of the
seasonal template estimated from the state curve (T1). See docs/TIMING_MODEL.md.

Modules: :mod:`.signals` (data), :mod:`.features` (feature blocks and shifts),
:mod:`.model` (year-specific-offset DM for T1, zero-coefficient check),
:mod:`.run` (fold driver), :mod:`.report` (tables and figures).
"""
