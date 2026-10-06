"""
Conformal routing logic for DSSE Stage-1.

Maps ICP prediction sets to routing decisions.
"""

# Class labels
BENIGN = 0
MALICIOUS = 1

# Routing decisions
FINAL_BENIGN = "FINAL_BENIGN"
FINAL_MALICIOUS = "FINAL_MALICIOUS"
NEEDS_STAGE2 = "NEEDS_STAGE2"
EMPTY_SET = "EMPTY_SET"


def route_prediction_set(prediction_set: list) -> str:
    """Map a conformal prediction set to a routing decision.

    Parameters
    ----------
    prediction_set : list of int
        Classes in the prediction set.  Valid members: 0 (benign), 1 (malicious).

    Returns
    -------
    str
        One of FINAL_BENIGN, FINAL_MALICIOUS, NEEDS_STAGE2.
    """
    s = set(prediction_set)

    if s == {BENIGN}:
        return FINAL_BENIGN

    if s == {MALICIOUS}:
        return FINAL_MALICIOUS

    if s == {BENIGN, MALICIOUS}:
        return NEEDS_STAGE2

    if len(s) == 0:
        return EMPTY_SET

    return NEEDS_STAGE2
