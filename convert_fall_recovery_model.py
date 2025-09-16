import torch

checkpoint = None
filename = "logs/T1_fall_recovery_pretrain.pth"

print(f"Loading model from {checkpoint}")
model_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

logstd = model_dict["model"]["logstd"]
model_dict["model"]["logstd"] = torch.zeros(1, 12)
model_dict["model"]["logstd"][:, torch.cat((torch.arange(0, 5), torch.arange(6, 9), torch.arange(10, 12)))] = logstd

critic_0_weight = model_dict["model"]["critic.0.weight"]
model_dict["model"]["critic.0.weight"] = torch.zeros(256, 56)
model_dict["model"]["critic.0.weight"][:, torch.cat((torch.arange(0, 11), torch.arange(12, 15), torch.arange(16, 23), torch.arange(24, 27), torch.arange(28, 35), torch.arange(36, 39), torch.arange(40, 56)))] = critic_0_weight

actor_0_weight = model_dict["model"]["actor.0.weight"]
model_dict["model"]["actor.0.weight"] = torch.zeros(256, 106)
model_dict["model"]["actor.0.weight"][:, torch.cat((torch.arange(0, 11), torch.arange(12, 15), torch.arange(16, 23), torch.arange(24, 27), torch.arange(28, 35), torch.arange(36, 39), torch.arange(40, 106)))] = actor_0_weight

actor_6_weight = model_dict["model"]["actor.6.weight"]
model_dict["model"]["actor.6.weight"] = torch.zeros(12, 128)
model_dict["model"]["actor.6.weight"][torch.cat((torch.arange(0, 5), torch.arange(6, 9), torch.arange(10, 12))), :] = actor_6_weight

actor_6_bias = model_dict["model"]["actor.6.bias"]
model_dict["model"]["actor.6.bias"] = torch.zeros(12)
model_dict["model"]["actor.6.bias"][torch.cat((torch.arange(0, 5), torch.arange(6, 9), torch.arange(10, 12)))] = actor_6_bias

adaptation_module_0_weight = model_dict["model"]["adaptation_module.0.weight"].view(1024, 50, 36)
model_dict["model"]["adaptation_module.0.weight"] = torch.zeros(1024, 50, 42)
model_dict["model"]["adaptation_module.0.weight"][:, :, torch.cat((torch.arange(0, 11), torch.arange(12, 15), torch.arange(16, 23), torch.arange(24, 27), torch.arange(28, 35), torch.arange(36, 39), torch.arange(40, 42)))] = adaptation_module_0_weight
model_dict["model"]["adaptation_module.0.weight"] = model_dict["model"]["adaptation_module.0.weight"].view(1024, -1)

torch.save({"model": model_dict["model"]}, filename)
print(f"Successfully written to {filename}")
