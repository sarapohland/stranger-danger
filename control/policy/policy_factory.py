from control.policy.sarl import SARL
from control.policy.uncertain_sarl import UNCERTAIN_SARL
from simulation.envs.policy.policy_factory import policy_factory

policy_factory['sarl'] = SARL
policy_factory['uncertain_sarl'] = UNCERTAIN_SARL
