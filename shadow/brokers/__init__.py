"""Broker clients for Shadow -- read-only for both. Neither
`shadow.brokers.tastytrade` nor `shadow.brokers.ibkr` (nor anything else
under `shadow/`) contains code capable of placing, modifying, or cancelling
a real order. See each module's own docstring for the specific
defense-in-depth reasoning.
"""
