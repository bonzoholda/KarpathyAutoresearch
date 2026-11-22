"""Standalone funding-percentile signal module.

Non-invasive: publishes through the existing /webhook/strategy door only.
Never arms live positions - it emits inert shadow rows with unreachable RSI
thresholds and carries the real signal in the diagnostic research payload.
"""
