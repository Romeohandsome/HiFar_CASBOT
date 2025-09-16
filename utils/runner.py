import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.wrapper import StackObservationWrapper
from utils.model import *
from utils.buffer import ExperienceBuffer
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder
from envs import *


class Runner:

    def __init__(self, test=False):
        self.test = test
        self.get_args()
        self.update_cfg_from_args()
        self.set_seed()
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)
        self.env = StackObservationWrapper(self.env, self.cfg["runner"]["num_stack"])

        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.model = DenoisingRMA(self.env.num_actions, self.env.num_obs, self.env.num_stack, self.env.num_privileged_obs, self.cfg["algorithm"]["num_embedding"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.load()

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("stacked_obses", (self.env.num_stack, self.env.num_obs))
        self.buffer.add_buffer("mirrored_obses", (self.env.num_obs,))
        self.buffer.add_buffer("mirrored_privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("mirrored_stacked_obses", (self.env.num_stack, self.env.num_obs))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        self.args = parser.parse_args()

    def update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        if not self.test:
            self.cfg["viewer"]["record_video"] = False

    def set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def load(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self.env.curriculum_prob = model_dict["curriculum"]
        except Exception as e:
            print(f"Failed to load curriculum: {e}")
        try:
            self.optimizer.load_state_dict(model_dict["optimizer"])
        except Exception as e:
            print(f"Failed to load optimizer: {e}")

    def train(self):
        self.recorder = Recorder(self.cfg)
        obs, rew, done, infos = self.env.reset()
        obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        stacked_obs = infos["stacked_obs"].to(self.device)

        with tqdm(total=self.cfg["basic"]["max_iterations"], desc="Training Progress") as pbar_outer:
            for it in range(self.cfg["basic"]["max_iterations"]):
                with tqdm(total=self.cfg["runner"]["horizon_length"], desc=f"Epoch {it + 1}") as pbar_inner:
                    for n in range(self.cfg["runner"]["horizon_length"]):
                        mirrored_obs = self.env.mirror_obs(obs)
                        mirrored_privileged_obs = self.env.mirror_priv(privileged_obs)
                        mirrored_stacked_obs = self.env.mirror_obs(stacked_obs)
                        self.buffer.update_data("obses", n, obs)
                        self.buffer.update_data("privileged_obses", n, privileged_obs)
                        self.buffer.update_data("stacked_obses", n, stacked_obs)
                        self.buffer.update_data("mirrored_obses", n, mirrored_obs)
                        self.buffer.update_data("mirrored_privileged_obses", n, mirrored_privileged_obs)
                        self.buffer.update_data("mirrored_stacked_obses", n, mirrored_stacked_obs)
                        
                        with torch.no_grad():
                            dist, _ = self.model.act(obs, stacked_obs=stacked_obs)
                            mirrored_dist, _ = self.model.act(mirrored_obs, stacked_obs=mirrored_stacked_obs)
                            
                            act = 0.5 * (dist.loc + self.env.mirror_act(mirrored_dist.loc)) + dist.scale * torch.randn_like(dist.loc)

                        obs, rew, done, infos = self.env.step(act)
                        obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                        privileged_obs = infos["privileged_obs"].to(self.device)
                        stacked_obs = infos["stacked_obs"].to(self.device)
                        self.buffer.update_data("actions", n, act)
                        self.buffer.update_data("rewards", n, rew)
                        self.buffer.update_data("dones", n, done)
                        self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                        
                        ep_info = {"reward": rew}
                        ep_info.update(infos["rew_terms"])
                        self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))
                        
                        pbar_inner.update(1)
                    self.last_obs = obs
                    self.last_privileged_obs = privileged_obs

                    self.denoising_step(it)

                    if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                        self.recorder.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, it + 1)

                pbar_outer.update(1)
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))

    def play(self):
        obs, rew, done, infos = self.env.reset()
        obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        stacked_obs = infos["stacked_obs"].to(self.device)
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        while True:
            with torch.no_grad():
                mirrored_obs = self.env.mirror_obs(obs)
                mirrored_privileged_obs = self.env.mirror_priv(privileged_obs)
                mirrored_stacked_obs = self.env.mirror_obs(stacked_obs)
                dist, _ = self.model.act(obs, stacked_obs=stacked_obs)
                mirrored_dist, _ = self.model.act(mirrored_obs, stacked_obs=mirrored_stacked_obs)
                act = 0.5 * (dist.loc + self.env.mirror_act(mirrored_dist.loc))
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                stacked_obs = infos["stacked_obs"].to(self.device)
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as self.writer:
                        for frame in self.env.camera_frames:
                            self.writer.append_data(frame)
                    if self.interrupt:
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True

    def denoising_step(self, it):
        with torch.no_grad():
            old_dist, _ = self.model.act(self.buffer["obses"], stacked_obs=self.buffer["stacked_obses"])
            mirrored_old_dist, _ = self.model.act(self.buffer["mirrored_obses"], stacked_obs=self.buffer["mirrored_stacked_obses"])
            sym_old_dist = torch.distributions.Normal(0.5 * (old_dist.loc + self.env.mirror_act(mirrored_old_dist.loc)), old_dist.scale)
            old_actions_log_prob = sym_old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

        mean_value_loss = 0
        mean_actor_loss = 0
        mean_bound_loss = 0
        mean_denoising_loss = 0
        mean_embedding_norm_loss = 0
        mean_entropy = 0
        mean_symmetric_loss = 0
        for n in range(self.cfg["runner"]["mini_epochs"]):
            values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
            last_values = self.model.est_value(self.last_obs, self.last_privileged_obs)
            with torch.no_grad():
                self.buffer["rewards"][self.buffer["time_outs"]] = values[self.buffer["time_outs"]]
                advantages = discount_values(self.buffer["rewards"], self.buffer["dones"] | self.buffer["time_outs"], values, last_values, self.cfg["algorithm"]["gamma"], self.cfg["algorithm"]["lam"])
                returns = values + advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            value_loss = F.mse_loss(values, returns)

            dist, embedding, privileged_obs_est = self.model.act(self.buffer["obses"], stacked_obs=self.buffer["stacked_obses"], decoder=True)
            mirrored_dist, mirrored_embedding, mirrored_privileged_obs_est = self.model.act(self.buffer["mirrored_obses"], stacked_obs=self.buffer["mirrored_stacked_obses"], decoder=True)
            mirrored_act = self.env.mirror_act(mirrored_dist.loc)
            symmetric_dist = torch.distributions.Normal(0.5 * (dist.loc + mirrored_act), dist.scale)
            actions_log_prob = symmetric_dist.log_prob(self.buffer["actions"]).sum(dim=-1)
            actor_loss = surrogate_loss(old_actions_log_prob, actions_log_prob, advantages)

            bound_loss = torch.clip(symmetric_dist.loc - 1.0, min=0.0).square().mean() + torch.clip(symmetric_dist.loc + 1.0, max=0.0).square().mean()

            denoising_loss = F.mse_loss(privileged_obs_est, self.buffer["privileged_obses"]) + F.mse_loss(mirrored_privileged_obs_est, self.buffer["mirrored_privileged_obses"])

            entropy = dist.entropy().sum(dim=-1)

            embedding_norm_loss = torch.clip(embedding.square().mean(dim=-1) - 1.0, min=0.0).square().mean() + torch.clip(mirrored_embedding.square().mean(dim=-1) - 1.0, min=0.0).square().mean()

            symmetric_loss = F.mse_loss(dist.loc, mirrored_act)

            loss = (
                value_loss
                + actor_loss
                + self.cfg["algorithm"]["bound_coef"] * bound_loss
                + self.cfg["algorithm"]["denoising_coef"] * denoising_loss
                + self.cfg["algorithm"]["embedding_norm_coef"] * embedding_norm_loss
                + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                + self.cfg["algorithm"]["symmetric_coef"] * symmetric_loss
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.ac_parameters(), 1.0)
            self.optimizer.step()

            with torch.inference_mode():
                kl = torch.sum(torch.log(dist.scale / old_dist.scale) + 0.5 * (torch.square(old_dist.scale) + torch.square(dist.loc - old_dist.loc)) / torch.square(dist.scale) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            mean_value_loss += value_loss.item()
            mean_actor_loss += actor_loss.item()
            mean_bound_loss += bound_loss.item()
            mean_denoising_loss += denoising_loss.item()
            mean_embedding_norm_loss += embedding_norm_loss.item()
            mean_entropy += entropy.mean()
            mean_symmetric_loss += symmetric_loss.item()
        mean_value_loss /= self.cfg["runner"]["mini_epochs"]
        mean_actor_loss /= self.cfg["runner"]["mini_epochs"]
        mean_bound_loss /= self.cfg["runner"]["mini_epochs"]
        mean_denoising_loss /= self.cfg["runner"]["mini_epochs"]
        mean_embedding_norm_loss /= self.cfg["runner"]["mini_epochs"]
        mean_entropy /= self.cfg["runner"]["mini_epochs"]
        mean_symmetric_loss /= self.cfg["runner"]["mini_epochs"]
        self.recorder.record_statistics(
            {
                "value_loss": mean_value_loss,
                "actor_loss": mean_actor_loss,
                "bound_loss": mean_bound_loss,
                "denoising_loss": mean_denoising_loss,
                "embedding_norm_loss": mean_embedding_norm_loss,
                "entropy": mean_entropy,
                "symmetric_loss": mean_symmetric_loss,
                "kl_mean": kl_mean,
                "lr": self.learning_rate,
            },
            it,
        )
