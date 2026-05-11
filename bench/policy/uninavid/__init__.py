"""UniNaVid policy package for OmniNavBench.

This package provides HTTP client-server integration for the Uni-NaVid
navigation model (RSS 2025).

Usage:
    # Start the server in Uni-NaVid conda environment:
    # python -m bench.policy.uninavid.uninavid_server --model_path ... --port 8000

    # Use the policy in OmniNavBench:
    from bench.policy.uninavid import UniNaVidHTTPPolicy
    policy = UniNaVidHTTPPolicy(server_url="http://localhost:8000")
"""

from .uninavid_http_policy import UniNaVidHTTPPolicy

__all__ = ["UniNaVidHTTPPolicy"]
