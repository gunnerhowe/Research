r"""Event -> modeled energy accounting.

The paper reports EVENT COUNTS and MODELED energy, never measured watts.
Cost model follows the delta-network literature's use of Horowitz's ISSCC 2014
"Computing's energy problem" 45nm numbers:

    FP32 multiply 3.7 pJ + FP32 add 0.9 pJ -> E_MAC = 4.6 pJ
    32-bit SRAM read (32 KB array)          E_SRAM = 5.0 pJ
    32-bit DRAM access                      E_DRAM = 640 pJ (not used: weights
                                            assumed SRAM-resident, stated)

One event = one transmitted vector component = fanout MACs + fanout weight
fetches from SRAM + one activation-anchor read-modify-write (2 SRAM ops).
Dense cost = every component fires every step.
"""

E_MAC_PJ = 4.6
E_SRAM_PJ = 5.0


def layer_energy_pj(events, fanout):
    """Modeled energy of `events` transmitted components with given fan-out."""
    return events * (fanout * (E_MAC_PJ + E_SRAM_PJ) + 2.0 * E_SRAM_PJ)


def gru_stats_energy_pj(stats):
    """Total modeled energy of a DeltaGRU forward from its stats list."""
    total = 0.0
    for s in stats:
        total += layer_energy_pj(s["events_x"], s["fanout_x"])
        total += layer_energy_pj(s["events_h"], s["fanout_h"])
    return total


def gru_stats_dense_energy_pj(stats):
    """Same model, every component firing every step (dense reference)."""
    total = 0.0
    for s in stats:
        total += layer_energy_pj(s["dense_x"], s["fanout_x"])
        total += layer_energy_pj(s["dense_h"], s["fanout_h"])
    return total


def gru_stats_event_rate(stats):
    """Events per component-step, pooled over layers (the paper's 'events/step'
    is this rate times the total component count)."""
    ev = sum(s["events_x"] + s["events_h"] for s in stats)
    dense = sum(s["dense_x"] + s["dense_h"] for s in stats)
    return ev / dense
