"""Evaluation and monitoring layer for the net_new pipeline.

Deliberately kept outside net_new/ so nothing here affects
net_new.pipeline.config()'s implementation_hash — this package can be edited
freely without invalidating cache-mode replay of the supplied baseline.
"""
