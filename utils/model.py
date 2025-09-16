import torch
import torch.nn.functional as F


class RMA(torch.nn.Module):
    def __init__(self, num_act, num_obs, obs_stacking, num_privileged_obs, num_embedding):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_embedding, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.privileged_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_privileged_obs, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_embedding),
        )
        self.adaptation_module = torch.nn.Sequential(
            torch.nn.Linear(num_obs * obs_stacking, 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_embedding),
        )
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=-2.0), requires_grad=True)

    def forward(self):
        raise NotImplementedError

    def act(self, obs, privileged_obs=None, stacked_obs=None):
        if privileged_obs is not None:
            embedding = self.privileged_encoder(privileged_obs)
        if stacked_obs is not None:
            embedding = self.adaptation_module(stacked_obs.flatten(start_dim=-2))
        act_input = torch.cat((obs, embedding), dim=-1)
        action_mean = self.actor(act_input)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        return dist, embedding

    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)

    def ac_parameters(self):
        for p in self.critic.parameters():
            yield p
        for p in self.actor.parameters():
            yield p
        for p in self.privileged_encoder.parameters():
            yield p
        yield self.logstd

    def adapt_parameters(self):
        for p in self.adaptation_module.parameters():
            yield p


class DenoisingRMA(RMA):

    def __init__(self, num_act, num_obs, num_stack, num_privileged_obs, num_embedding):
        super().__init__(num_act, num_obs, num_stack, num_privileged_obs, num_embedding)
        self.privileged_decoder = torch.nn.Sequential(
            torch.nn.Linear(num_embedding, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_privileged_obs),
        )

    def act(self, obs, stacked_obs, decoder=False):
        embedding = self.adaptation_module(stacked_obs.flatten(start_dim=-2))
        act_input = torch.cat((obs, embedding), dim=-1)
        action_mean = self.actor(act_input)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        if decoder:
            privileged_obs_est = self.privileged_decoder(embedding)
            return dist, embedding, privileged_obs_est
        else:
            return dist, embedding
