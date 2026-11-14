"""
Configuration and Constants for the AML Graph Investigation Engine.
All thresholds, time windows, and risk scoring weights are centrally defined here.

IMPORTANT DISCLAIMER:
- These thresholds (e.g., Fan-In >= 4 senders in 48h, Fan-Out >= 4 destinations in 48h)
  are PROTOTYPE INVESTIGATION PARAMETERS and DO NOT represent official AML regulatory thresholds.
- Risk scores (0-100) are PROTOTYPE INVESTIGATION RISK SCORES and DO NOT represent
  official AML regulatory scores or real-world probabilities of money laundering.
"""

# Pattern Detection Thresholds (Configurable Prototype Parameters)
FAN_IN_MIN_SENDERS = 4
FAN_IN_TIME_WINDOW_HOURS = 48

FAN_OUT_MIN_DESTINATIONS = 4
FAN_OUT_TIME_WINDOW_HOURS = 48

MIN_CYCLE_LENGTH = 3
MAX_CYCLE_LENGTH = 6

MAX_MULTIHOP_HOPS = 6
MAX_MULTIHOP_TIME_HOURS = 72

SCATTER_GATHER_TIME_WINDOW_HOURS = 72

# Risk Scoring Weights (Transparent Prototype Model)
RISK_WEIGHTS = {
    "graph_score": 0.30,          # Topological risk (fan-in/out, cycle membership)
    "ml_score": 0.30,             # Pre-trained Random Forest probability & Isolation Forest anomaly
    "velocity_score": 0.15,       # Rapid transaction frequency & short intervals
    "network_complexity": 0.15,   # Hops, unique counterparties, degree
    "cross_bank_context": 0.10,   # Cross-bank / cross-currency movement (contextual indicator only)
}

# Risk Level Tier Boundaries (0 - 100)
RISK_TIERS = {
    "CRITICAL": 81,
    "HIGH": 61,
    "MEDIUM": 31,
    "LOW": 0,
}
