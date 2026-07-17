"""LangSmith evaluation harness for the payer claims agent.

This package is the "test" leg of the observe -> test -> improve loop:

* ``dataset``    — canonical COB scenarios with reference outputs, uploadable
                   to LangSmith as an offline eval dataset.
* ``evaluators`` — deterministic code evaluators for the auditable core, plus
                   an LLM-as-judge for the LLM intake node.
* ``run_experiment`` — wires the graph as an eval target and runs experiments.
"""
