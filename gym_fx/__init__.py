from gym.envs.registration import register
register(
    id='fx-v0',
    entry_point='gym_fx.envs:FxEnv',
)

