DOMAIN_CONFIGS = {
    "coding": {
        "drift_threshold": 0.4,
        "loop_threshold": 3,
        "token_multiplier": 3.0,
    },
    "research": {
        "drift_threshold": 0.35,
        "loop_threshold": 3,
        "token_multiplier": 3.5,
    },
    "medical": {
        "drift_threshold": 0.45,
        "loop_threshold": 2,
        "token_multiplier": 4.0,
    },
}

DEFAULT_DOMAIN = "coding"


def get_config(domain: str = DEFAULT_DOMAIN) -> dict:
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(DOMAIN_CONFIGS.keys())}")
    return DOMAIN_CONFIGS[domain]
