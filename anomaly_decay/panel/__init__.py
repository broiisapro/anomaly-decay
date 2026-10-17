"""Panel construction: window classification and merging returns with metadata and factors.

Panel returns are **percent per month** (OSAP and Ken French use the same units, so no
conversion happens anywhere). Windows follow docs/architecture.md §4.6: ``in_sample``
covers January ``SampleStartYear`` through December ``SampleEndYear``,
``post_publication`` starts January ``Year+1`` (default) or January ``Year`` (alt),
``post_sample`` fills the gap and may be empty, and ``in_sample`` wins when publication
would start inside the sample. Pre-sample months are dropped at build time, so every
panel row is in exactly one window.
"""
