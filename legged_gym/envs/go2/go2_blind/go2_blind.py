from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq import Go2Dreamwaq


class GO2Blind(Go2Dreamwaq):
    """DreamWaQ-style blind baseline.

    The actor never receives an explicit terrain height map. It acts from the
    current proprioceptive observation and a latent inferred from proprioceptive
    history, while the critic and auxiliary VAE losses may use privileged
    training signals.
    """

    pass
