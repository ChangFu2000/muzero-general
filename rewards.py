final_reward_only = {
    "attacker": {
        "reward": {
            "compromise_critical": 1,
        },
        "penalty": {
            "caught_by_defender": 0,
            "exceed_maximum_action": 0,
        },
    },
    "defender": {
        "reward": {
            "detect_attacker": 1,
        },
    },
}

detail_reward = {
    "attacker": {
        "cost": {
            "scan": 0,
            "exploit": 0,
        },
        "reward": {
            "compromise_noncritical": 1000,
            "compromise_critical": 5000,
        },
        "penalty": {
            "exploit_nonexist_service": -10,
            "access_unreachable_machine": -30,
            "compromise_again": -20,
            "exceed_maximum_action": -100,
            "compromise_unscanned_machine": -10,
            "caught_by_defender": -100,
        },
    },
    "defender": {
        "cost": {
            "defend": -5,
        },
        "reward": {
            "detect_attacker": 200,
        },
        "penalty": {
            "defend_nonexist_service": -10,
        },
    },
}