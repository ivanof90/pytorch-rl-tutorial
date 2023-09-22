from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

#Define hyperparameters
device = "cpu" if not torch.has_cuda else "cuda:0"
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0 #gradient clipping

#Data collection parameters
frame_skip = 1
frames_per_batch = 1000 // frame_skip
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000 // frame_skip

#PPO Parameters
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99 # discount parameter, if the value es near to 1, the inmediate rewards are more importatns
lmbda = 0.95 # bias/variance trade-off for the Advantage function in PPO, which calculates the TD value (The critic).
entropy_eps = 1e-4 # is a hyperparameter used to control the amount of entropy regularization applied
# during the optimization process. Entropy regularization is a technique used in reinforcement learning
# to encourage exploration and prevent policies from becoming too deterministic.

#define the envyroment
base_env = GymEnv("InvertedDoublePendulum-v4", device=device, frame_skip=frame_skip, render_mode="human")

env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]), # in keys, refers to tensordict.TensorDict
        DoubleToFloat(
            in_keys=["observation"],
        ),#will convert double entries to single-precision numbers, ready to be read by the policy
        StepCounter(), # will be used to count the steps before the environment is terminated. We will use this measure as a supplementary measure of performance.
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0) # init the normalization parameters
print("normalization constant shape:", env.transform[0].loc.shape)

#check the most important features of our enviroments
print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("done_spec:", env.done_spec)
print("action_spec:", env.action_spec)
#print("state_spec:", env.state_spec)

check_env_specs(env) # if check_env_specs succeeded! if that true, the enviroment is ready to go...

rollout = env.rollout(4) #execute N steps with random actions...
print("rollout of three steps:", rollout) # The "next" entry points to the data coming after the current step. In most cases, the "next"" data at time t matches the data at t+1, but this may not be the case if we are using some specific transformations
print("Shape of the rollout TensorDict:", rollout.batch_size) #  matches the number of steps we ran it for

#Now, we create the actor network (the policy) we use a Tanh activation fuction due the type of action space in the
#inverted pendelum (continuos), so it will create an normilized output (-1,1) indicating the action
# (a force maden on the pendelum, according to the docs is (-3,3)). It will return the required parameters to scale out
# to the boundiers of the action space (-3,3)

#During training, the actor's network outputs the mean and standard deviation (log variance)
# of the Gaussian distribution (this distrubution is builded in ProbabilisticActor) for -each action dimension-,
# that is the reason why we need to double the output against the action space size
#The NormalParamExtractor then processes these raw outputs to ensure the standard deviation is positive
# (by taking the exponent of the output) and creates a distribution with these extracted parameters (similar to softmax).

#By using a probability distribution to model the policy and extracting the distribution's parameters with
# the NormalParamExtractor, we can effectively handle continuous action spaces,
# enabling the agent to learn robust policies and perform well in such environments.
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device), # it's like torch.nn.Linear but it doesnt need the number of input layer
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device), #its necesary to duplitate the parameters of the action space in this framework, with a continous output
    #for each action, we need the loc and the scale
    NormalParamExtractor(), # to extract the location (or mean) and the scale (or standard desviation == log(variance)),
    # The NormalParamExtractor provides a way to create a stochastic policy that is differentiable.
    # Differentiability is crucial for policy gradient methods like PPO, as it allows us to compute gradients and update the policy in a stable and efficient manner through backpropagation
    # Instead of directly predicting the action values, we predict the parameters (mean and standard deviation) of the Gaussian distribution
)


policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
) # using TensorDict, the policy can interact with the enviroment.


policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.minimum,
        "max": env.action_spec.space.maximum,
    },
    return_log_prob=True,  # we'll need the log-prob for the numerator of the importance weights

) # Helps to create the stochastic policy. It will recieve the arrays of loc and scale, and creates the distribution
#using the TanhNormal, which creates a normal distribution but the loc (or mean) is scaled using:
#loc = tanh(loc/upscale) * upscale, where upscale is parameter of TanhNormal (by default is 5.0) and clipped by min and max
#*************
#The value for the action is amoung the range of this distribution. At each step,  the scale(sd) is reduced, so the policy gets
#more accuracy during the training, selecting the best value for the current state

#Now we declare the critic/value network and module
value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)


#testing the modules
print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
) # see MultiSyncDataCollector and MultiaSyncDataCollector for more advance collectors
# A collector reset an environment, compute an action given the latest observation,
# execute a step in the environment,
# and repeat the last two steps until the environment signals a stop.
# tensordict.TensorDict instances with a total number of elements that will match frames_per_batch

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)# just to storage the frames to read per batch. This is still online learning

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
) # to calculate the advantage funcion


loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
) #lost function for PPO

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)# LR Scheduler, the value of LR varies in a cyclical way according to the function defined in: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
#the minimum vale of LR in this case is 0.0. It needs the total number of epochs: total_frames // frames_per_batch (integer division)

logs = defaultdict(list)
pbar = tqdm(total=total_frames * frame_skip)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        with torch.no_grad():
            advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optim step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel() * frame_skip)
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our env horizon).
        # The ``rollout`` method of the env can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()