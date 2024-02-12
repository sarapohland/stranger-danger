class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'


class Danger(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close'


class HumanCollision(object):
    def __init__(self, position, human=None):
        self.coll_pos = position
        self.human = human

    def __str__(self):
        return 'Human collision'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''
