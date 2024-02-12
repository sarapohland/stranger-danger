from simulation.envs.policy.orca import ORCA
from simulation.envs.policy.linear import Linear


def none_policy():
    return None

policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
