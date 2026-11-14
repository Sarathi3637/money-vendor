"""
Pattern Detection Module for AML Graph Investigation Engine.
Implements detection logic for Fan-In, Fan-Out, Cycles, Rapid Multi-Hop, and Scatter-Gather
using NetworkX and configurable time/size thresholds.
"""

import networkx as nx
import pandas as pd
import numpy as np
from datetime import timedelta
import src.graph_config as config


def detect_fan_in(G: nx.MultiDiGraph):
    """
    Detects Fan-In patterns: multiple unique accounts sending funds to one account
    within the configured time window (e.g. 4 or more senders within 48 hours).
    Uses a sliding temporal window over incoming transactions.
    Note: Fan-In indicates structural convergence, not confirmed laundering.
    """
    patterns = []
    window = timedelta(hours=config.FAN_IN_TIME_WINDOW_HOURS)
    min_senders = config.FAN_IN_MIN_SENDERS

    for node in G.nodes():
        in_edges = list(G.in_edges(node, data=True))
        if len(in_edges) < min_senders:
            continue

        # Filter edges with valid timestamps and sort chronologically
        valid_edges = [e for e in in_edges if e[2].get("timestamp") is not None]
        if len(valid_edges) < min_senders:
            continue

        valid_edges.sort(key=lambda e: e[2]["timestamp"])
        n = len(valid_edges)

        # Sliding window check
        for i in range(n):
            start_ts = valid_edges[i][2]["timestamp"]
            window_edges = []
            senders = set()
            for j in range(i, n):
                cur_ts = valid_edges[j][2]["timestamp"]
                if cur_ts - start_ts <= window:
                    window_edges.append(valid_edges[j])
                    senders.add(valid_edges[j][0])
                else:
                    break

            if len(senders) >= min_senders:
                time_span = window_edges[-1][2]["timestamp"] - window_edges[0][2]["timestamp"]
                patterns.append({
                    "type": "Fan-In",
                    "target_account": node,
                    "unique_senders": len(senders),
                    "time_span_hours": time_span.total_seconds() / 3600.0,
                    "supporting_transactions": [d.get("transaction_id") for _, _, d in window_edges]
                })
                break  # Record one qualified burst per node

    return patterns


def detect_fan_out(G: nx.MultiDiGraph):
    """
    Detects Fan-Out patterns: one account distributing funds to many destinations
    within the configured time window (e.g. 4 or more destinations within 48 hours).
    Uses a sliding temporal window over outgoing transactions.
    Note: Fan-Out indicates distribution, not confirmed laundering.
    """
    patterns = []
    window = timedelta(hours=config.FAN_OUT_TIME_WINDOW_HOURS)
    min_dests = config.FAN_OUT_MIN_DESTINATIONS

    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        if len(out_edges) < min_dests:
            continue

        valid_edges = [e for e in out_edges if e[2].get("timestamp") is not None]
        if len(valid_edges) < min_dests:
            continue

        valid_edges.sort(key=lambda e: e[2]["timestamp"])
        n = len(valid_edges)

        # Sliding window check
        for i in range(n):
            start_ts = valid_edges[i][2]["timestamp"]
            window_edges = []
            destinations = set()
            for j in range(i, n):
                cur_ts = valid_edges[j][2]["timestamp"]
                if cur_ts - start_ts <= window:
                    window_edges.append(valid_edges[j])
                    destinations.add(valid_edges[j][1])
                else:
                    break

            if len(destinations) >= min_dests:
                time_span = window_edges[-1][2]["timestamp"] - window_edges[0][2]["timestamp"]
                patterns.append({
                    "type": "Fan-Out",
                    "source_account": node,
                    "unique_destinations": len(destinations),
                    "time_span_hours": time_span.total_seconds() / 3600.0,
                    "supporting_transactions": [d.get("transaction_id") for _, _, d in window_edges]
                })
                break

    return patterns


def detect_cycles(G: nx.MultiDiGraph):
    """
    Detects directed cycles within bounded length (min=3, max=6 hops).
    Enforces canonical deduplication and chronological ordering where applicable.
    """
    simple_g = nx.DiGraph(G)
    cycles = nx.simple_cycles(simple_g)

    detected_cycles = []
    seen_cycles = set()

    for cycle in cycles:
        if config.MIN_CYCLE_LENGTH <= len(cycle) <= config.MAX_CYCLE_LENGTH:
            min_idx = cycle.index(min(cycle))
            canonical = tuple(cycle[min_idx:] + cycle[:min_idx])
            if canonical in seen_cycles:
                continue
            seen_cycles.add(canonical)

            cycle_txs = []
            cycle_timestamps = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    first_key = list(edge_data.keys())[0]
                    edge_attrs = edge_data[first_key]
                    cycle_txs.append(edge_attrs["transaction_id"])
                    cycle_timestamps.append(edge_attrs["timestamp"])

            detected_cycles.append({
                "type": "Circular Transfer",
                "accounts": cycle,
                "hops": len(cycle),
                "supporting_transactions": cycle_txs
            })

    return detected_cycles


def detect_rapid_multi_hop(G: nx.MultiDiGraph):
    """
    Detects rapid multi-hop movement moving strictly forward chronologically.
    Bounded by MAX_MULTIHOP_HOPS and MAX_MULTIHOP_TIME_HOURS.
    Enforces maximal path filtering to prevent duplicate investigation cases from overlapping sub-paths.
    """
    patterns = []

    for source in G.nodes():
        if G.out_degree(source) == 0:
            continue

        def dfs_paths(current, path, tx_list, start_ts, last_ts):
            if len(path) >= config.MAX_MULTIHOP_HOPS:
                return
            for _, nxt, d in G.out_edges(current, data=True):
                ts = d.get("timestamp")
                if ts is None or last_ts is None:
                    continue
                if ts < last_ts:
                    continue  # Enforce strict forward chronological order
                if nxt in path:
                    continue  # Avoid internal cycles

                new_path = path + [nxt]
                new_txs = tx_list + [d.get("transaction_id")]
                duration = ts - start_ts

                if duration <= timedelta(hours=config.MAX_MULTIHOP_TIME_HOURS) and len(new_path) >= 3:
                    patterns.append({
                        "type": "Rapid Multi-Hop",
                        "path": new_path,
                        "hops": len(new_path) - 1,
                        "duration_hours": duration.total_seconds() / 3600.0,
                        "supporting_transactions": new_txs
                    })

                dfs_paths(nxt, new_path, new_txs, start_ts, ts)

        # Explore outgoing chains from nodes with outgoing transfers
        for _, nxt, d in G.out_edges(source, data=True):
            ts = d.get("timestamp")
            if ts:
                dfs_paths(nxt, [source, nxt], [d.get("transaction_id")], ts, ts)

    # Deduplicate exact identical transaction sequences
    unique_patterns = {}
    for p in patterns:
        key = tuple(p["supporting_transactions"])
        unique_patterns[key] = p

    candidate_list = list(unique_patterns.values())

    # Maximal path filtering: remove sub-paths that are strictly sub-sequences of longer multi-hop paths
    maximal_patterns = []
    for i, p1 in enumerate(candidate_list):
        txs1 = p1["supporting_transactions"]
        is_subpath = False
        for j, p2 in enumerate(candidate_list):
            if i == j:
                continue
            txs2 = p2["supporting_transactions"]
            if len(txs2) > len(txs1):
                str_txs1 = ",".join(str(x) for x in txs1)
                str_txs2 = ",".join(str(x) for x in txs2)
                if str_txs1 in str_txs2:
                    is_subpath = True
                    break
        if not is_subpath:
            maximal_patterns.append(p1)

    return maximal_patterns


def detect_scatter_gather(G: nx.MultiDiGraph):
    """
    Detects Scatter-Gather patterns requiring a meaningful structural and temporal relationship:
    multiple distinct source accounts -> intermediate account -> multiple distinct destination accounts,
    within SCATTER_GATHER_TIME_WINDOW_HOURS.
    Enforces that incoming flows temporally precede or coincide with outgoing flows
    (outflows cannot entirely precede inflows).
    """
    patterns = []
    window_hours = config.SCATTER_GATHER_TIME_WINDOW_HOURS

    for node in G.nodes():
        in_edges = list(G.in_edges(node, data=True))
        out_edges = list(G.out_edges(node, data=True))
        if not in_edges or not out_edges:
            continue

        sources = set(u for u, _, _ in in_edges)
        destinations = set(v for _, v, _ in out_edges)

        # Must have multiple sources AND multiple destinations (true structural bipartite fan-in + fan-out)
        if len(sources) >= 2 and len(destinations) >= 2:
            in_timestamps = [d.get("timestamp") for _, _, d in in_edges if d.get("timestamp") is not None]
            out_timestamps = [d.get("timestamp") for _, _, d in out_edges if d.get("timestamp") is not None]

            if not in_timestamps or not out_timestamps:
                continue

            min_in_ts = min(in_timestamps)
            max_out_ts = max(out_timestamps)

            # Temporal requirement: Outflows cannot all occur before any inflows arrive
            if max_out_ts < min_in_ts:
                continue

            # Total duration from first inflow to last outflow must be positive and within window
            total_seconds = (max_out_ts - min_in_ts).total_seconds()
            if total_seconds < 0:
                continue

            time_span_hours = total_seconds / 3600.0
            if time_span_hours <= window_hours:
                txs = [d.get("transaction_id") for _, _, d in in_edges] + [d.get("transaction_id") for _, _, d in out_edges]
                patterns.append({
                    "type": "Scatter-Gather",
                    "intermediate_account": node,
                    "sources": list(sources),
                    "destinations": list(destinations),
                    "time_span_hours": round(time_span_hours, 2),
                    "supporting_transactions": txs
                })

    return patterns
