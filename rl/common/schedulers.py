class PolynomialSchedule:
    def __init__(self, init_value, target_value, transition_steps, power=1.0) -> None:
        self.init_value = init_value
        self.target_value = target_value
        self.transition_steps = transition_steps
        self.power = power

    def __call__(self, step):
        value = (
            self.target_value
            + (self.init_value - self.target_value)
            * (1 - step / float(self.transition_steps)) ** self.power
        )
        return value if step < self.transition_steps else self.target_value


class ExponentialSchedule:
    def __init__(self, init_value, target_value, decay_rate=0.95) -> None:
        self.init_value = init_value
        self.target_value = target_value
        self.decay_rate = decay_rate

    def __call__(self, step):
        value = self.init_value * (self.decay_rate**step)
        return max(value, self.target_value)
