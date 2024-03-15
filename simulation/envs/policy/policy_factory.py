from simulation.envs.policy.orca import ORCA
from simulation.envs.policy.cadrl import CADRL
from simulation.envs.policy.linear import Linear


def none_policy():
    return None

policy_names = ['linear', 'orca', 'cadrl']

policy_factory = dict()
policy_factory['orca'] = ORCA
policy_factory['cadrl'] = CADRL
policy_factory['linear'] = Linear
policy_factory['none'] = none_policy
