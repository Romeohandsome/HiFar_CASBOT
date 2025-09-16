import torch


class StackObservationWrapper:

    def __init__(self, env, num_stack):
        self.env = env
        self.num_stack = num_stack
        self.stacked_obs = torch.zeros(self.env.num_envs, self.num_stack, self.env.num_obs, device=self.env.device)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        if name in {"env", "num_stack", "stacked_obs"}:
            super().__setattr__(name, value)
        elif name.startswith("_"):
            raise AttributeError(f"attempted to set missing private attribute '{name}'")
        else:
            setattr(self.env, name, value)

    def reset(self):
        obs, rew, done, infos = self.env.reset()
        self.stacked_obs[:] = obs.unsqueeze(1)
        infos.update({"stacked_obs": self.stacked_obs})
        return obs, rew, done, infos

    def step(self, action):
        obs, rew, done, infos = self.env.step(action)
        self.stacked_obs = torch.roll(self.stacked_obs, 1, dims=1)
        self.stacked_obs[done, :, :] = obs[done, :].unsqueeze(1)
        self.stacked_obs[:, 0, :] = obs
        infos.update({"stacked_obs": self.stacked_obs})
        return obs, rew, done, infos
