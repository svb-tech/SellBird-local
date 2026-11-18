"""Local, zero-cost multi-product sales agent built on top of Sellbird (formerly Harvey).

All language-model inference goes to a local Ollama server. There is no
cloud fallback anywhere in this package (see docs/COST_AND_DEPENDENCY_STATEMENT.md).
"""
__all__ = ["hardware", "llm", "store", "workflows", "embeddings", "netguard"]
