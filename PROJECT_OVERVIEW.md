# SentryNet Project Overview

## What This Project Is

SentryNet is a trust-aware multi-agent reinforcement learning system for autonomous drone border surveillance.

The core research question is whether a lightweight decentralized trust mechanism based on exponential moving averages can make cooperative drones more robust against adversarial communication attacks such as packet dropping and GPS or coordinate spoofing, without using cryptography.

The project is designed to compare three training systems under controlled conditions:

- System A: no packet loss, no trust mechanism
- System B: packet loss only, no trust mechanism
- System C: packet loss plus trust mechanism

The scientific claim is that System C should outperform System A and System B under adversarial communication conditions, especially at higher packet drop rates.

## Main Purpose

The project has two goals:

1. Build a realistic drone-border-surveillance simulation in which multiple drones cooperate to detect and capture an intruder.
2. Measure whether trust-aware message aggregation improves robustness when communication is unreliable or attacked.

In practical terms, the project asks:

- Can the drones still coordinate when messages are missing or spoofed?
- Can a simple trust score help them ignore bad messages?
- Does this improve capture rate and training stability under attack?

## High-Level Architecture

The system is organized into five layers:

1. Environment layer
2. Communication layer
3. Trust layer
4. Learning layer
5. Experiment and reporting layer

### 1. Environment Layer

The environment is implemented in [border_env.py](border_env.py).

It simulates:

- A 20 x 20 x 10 meter airspace
- Three hunter drones controlled by MAPPO
- One ground sensor controlled by a discrete policy
- One intruder drone that moves autonomously
- Optional PyBullet physics and rendering
- Domain randomization at each environment reset

The environment exposes observation and action spaces for each agent and produces per-agent rewards and termination signals.

### 2. Communication Layer

The communication path is:

- Each drone broadcasts intruder-related information
- The message enters the adversarial channel
- The channel may drop or spoof the message
- Trust scores are updated for each sender
- The remaining messages are aggregated using trust weights
- The aggregated message is inserted into the drone observation vector

This is the research core of the project.

### 3. Trust Layer

The trust mechanism lives in [trust_module.py](trust_module.py) and [trust_aggregator.py](trust_aggregator.py).

For each sender, the trust module maintains an EMA-like score that increases when received information is close to the truth and decays when messages are dropped.

The aggregator then computes a trust-weighted mean of all received messages.

### 4. Learning Layer

The MAPPO implementation is in [networks.py](networks.py), [rollout_buffer.py](rollout_buffer.py), and [mappo_trainer.py](mappo_trainer.py).

It uses:

- A shared actor network for the three drones
- A centralized value network for critic estimation
- PPO-style clipped policy optimization
- Generalized Advantage Estimation
- Mini-batch SGD over collected rollouts

### 5. Experiment and Reporting Layer

Experiment scripts:

- [train.py](train.py) trains systems A, B, and C
- [run_trained.py](run_trained.py) runs a saved checkpoint
- [evaluate.py](evaluate.py) runs the full evaluation sweep
- [plot_results.py](plot_results.py) generates the publication plots

## Codebase Map

### Core Simulation Files

- [border_env.py](border_env.py): environment, observations, actions, rewards, communication pipeline
- [adversarial_channel.py](adversarial_channel.py): packet drop and spoofing simulation
- [trust_module.py](trust_module.py): EMA trust score update
- [trust_aggregator.py](trust_aggregator.py): trust-weighted message aggregation

### Learning Files

- [networks.py](networks.py): policy and value networks
- [rollout_buffer.py](rollout_buffer.py): rollout storage and GAE
- [mappo_trainer.py](mappo_trainer.py): training loop, evaluation, checkpointing

### Execution Files

- [train.py](train.py): CLI entry point for training
- [evaluate.py](evaluate.py): run the full condition sweep and export CSV
- [plot_results.py](plot_results.py): create final figures
- [run_trained.py](run_trained.py): run a trained checkpoint in stats or visual mode

### Support Files

- [diagnostic_3d.py](diagnostic_3d.py): sanity checks for the environment and communication pipeline
- [tests/test_phase1_3d.py](tests/test_phase1_3d.py): automated tests for Phase 1 behavior
- [results/full_experiment.csv](results/full_experiment.csv): evaluation output
- [checkpoints](checkpoints): saved model weights

## Environment Design

### Agents

The environment contains four agents:

- drone_0
- drone_1
- drone_2
- sensor_0

The three drones are the learning agents. The sensor is a simple discrete agent that can raise an alert.

### World State

The environment tracks:

- drone positions
- drone velocities
- intruder position
- intruder velocity
- battery levels
- wind vector
- trust modules
- communication channel state
- aggregated message buffer

### Observation Space

Each drone receives a 20-dimensional float32 observation:

- [0:3] own position
- [3:6] own velocity
- [6:9] trust-aggregated intruder position
- [9:12] trust-aggregated intruder velocity
- [12] sensor alert
- [13:16] relative intruder position if in field of view and range, else zeros
- [16] battery state
- [17:20] wind vector

The sensor receives a 4-dimensional observation:

- detected flag
- noisy x, y, z coordinates

### Action Space

Drone actions:

- Box(3) in [-1, 1]
- Interpreted as normalized thrust or motion commands

Sensor actions:

- Discrete(2)
- 0 means idle
- 1 means trigger alert

### Episode Dynamics

Each episode begins with reset and domain randomization.

The environment may randomize:

- drone mass
- wind
- sensor noise
- intruder speed

At each step, the system advances physics, computes communications, updates trust, builds observations, and calculates rewards.

## Communication Data Flow

This is the most important flow in the project.

### Step-by-Step

1. Each drone produces a message containing intruder state information.
2. The message is passed through [adversarial_channel.py](adversarial_channel.py).
3. The channel may drop the message with probability `p_drop`.
4. If a message is not dropped, it may be spoofed with probability `p_spoof`.
5. The trust module updates sender trust using the received message and the ground-truth intruder state.
6. The trust aggregator computes a trust-weighted average of the received messages.
7. The aggregated result is injected into the drone observation vector.
8. The MAPPO policy uses the observation to choose the next movement action.

### Why This Matters

The drones do not communicate using encryption or authentication.
Instead, they try to infer which messages are reliable by tracking sender quality over time.

This is what makes the project lightweight and decentralized.

## Trust Mechanism

### Trust Update Logic

The trust score for each sender is updated using an EMA-style rule.

Conceptually:

- Good messages increase trust
- Dropped messages reduce trust
- Trust is clipped to the range [0, 1]

The update is designed so that trust can react quickly to attack while still smoothing random noise.

### Aggregation Logic

The aggregator combines messages using trust weights.

In simple form:

- Each received message is multiplied by its sender trust score
- The weighted messages are summed
- The sum is divided by total trust mass
- If all messages are dropped or all trust scores are zero, a zero vector is returned

This ensures bad or unreliable sources have less influence on the shared estimate.

## Learning Algorithm

The learning system is a MAPPO-style actor-critic setup.

### Policy Network

[networks.py](networks.py) defines `PolicyNet`.

It maps a 20-dimensional drone observation to a 3-dimensional action.

Structure:

- Linear(20 -> 128)
- Tanh
- Linear(128 -> 128)
- Tanh
- Linear(128 -> 3)
- Tanh output through the mean head
- Learnable log standard deviation parameter

It outputs a Gaussian action distribution and samples thrust actions.

### Value Network

[networks.py](networks.py) defines `ValueNet`.

It takes a centralized 60-dimensional state formed by concatenating the three drone observations.

Structure:

- Linear(60 -> 128)
- Tanh
- Linear(128 -> 128)
- Tanh
- Linear(128 -> 1)

This critic estimates the value of the joint drone state.

### Rollout Collection

[rollout_buffer.py](rollout_buffer.py) stores trajectories for all drones over a rollout window.

It stores:

- observations
- actions
- rewards
- values
- log probabilities
- done flags

After a rollout is collected, it computes advantages using Generalized Advantage Estimation.

### PPO Update

[mappo_trainer.py](mappo_trainer.py) uses PPO clipping to update the policy.

The update uses:

- policy ratio clipping
- value regression loss
- entropy bonus
- gradient clipping

This gives the agent stable policy improvement without large destructive updates.

## Training Loop

The main training flow is:

1. Reset environment
2. Collect a rollout of `n_steps`
3. Store trajectories in the buffer
4. Compute GAE advantages and returns
5. Update policy and value networks for several epochs
6. Log metrics
7. Periodically evaluate and save checkpoints
8. Repeat until `total_steps` is reached

### Shared Policy Setup

The three drones use one shared policy network.
This means they learn a common behavior policy rather than three separate policies.

That makes training more efficient and promotes coordination.

### Centralized Critic

The critic sees the concatenated state of all three drones.
This improves value estimation because it has more context than each drone's local observation.

## The Three Systems

The entire experiment hinges on making Systems A, B, and C identical except for two factors.

### System A

- `p_drop = 0.0`
- `use_trust = False`

This is the clean baseline.

### System B

- `p_drop = 0.2`
- `use_trust = False`

This measures the effect of packet loss alone.

### System C

- `p_drop = 0.2`
- `use_trust = True`

This measures the effect of trust-aware aggregation under packet loss.

### Experimental Invariant

The comparison is only valid if all three systems are identical in every other respect:

- same architecture
- same optimizer settings
- same rollout length
- same total training steps
- same seeds
- same evaluation protocol

Any other difference would invalidate the scientific claim.

## Reward Structure

The reward function encourages capture while discouraging unsafe or inefficient behavior.

The main components are:

- positive reward for capture
- time penalty
- energy drain penalty
- security failure penalty
- close-range bonus
- collision penalty
- coverage reward based on drone separation

The reward is designed to balance aggressiveness, safety, and coordination.

## Why This Project Is Research-Relevant

The project demonstrates a low-overhead alternative to cryptographic trust enforcement.

Why that matters:

- cryptography can be expensive on small embedded platforms
- communication attacks can break cooperative control even when the physics is correct
- a trust mechanism can adapt to unreliable senders without requiring secure channels

In other words, the project tests whether the team can make the swarm robust by learning who to believe, rather than by trying to secure every packet.

## How Evaluation Works

[evaluate.py](evaluate.py) runs the trained checkpoints across a sweep of conditions.

It evaluates:

- 3 systems
- 6 drop rates
- 3 seeds
- 200 deterministic episodes per condition

It writes the results to [results/full_experiment.csv](results/full_experiment.csv).

The main output metrics are:

- capture_rate
- mean_steps
- mean_reward
- mean_trust
- mean_battery

## How Plots Are Generated

[plot_results.py](plot_results.py) reads the full experiment CSV and generates publication figures.

The intended plots are:

1. Capture rate vs packet drop rate
2. Steps to capture vs packet drop rate
3. Trust score dynamics over one episode
4. Training reward curves

These plots are meant to show whether System C remains effective when communication becomes unreliable.

## How To Run The Project

### Train

```powershell
python train.py --system A --seed 0
python train.py --system B --seed 0
python train.py --system C --seed 0
python train.py --system all --seeds 0 1 2
```

### Evaluate

```powershell
python evaluate.py --system all --seeds 0 1 2 --episodes 200
```

### Plot

```powershell
python plot_results.py
```

### Run a Trained Checkpoint

```powershell
python run_trained.py --checkpoint checkpoints/system_A/final.pt --mode stats --episodes 20 --seed 0
```

### Visualize

```powershell
python run_trained.py --checkpoint checkpoints/system_A/final.pt --mode visual
```

## How To Verify The Project Is Working

A healthy run should show:

- `diagnostic_3d.py` passing its checks
- tests in `tests/test_phase1_3d.py` passing or at least highlighting only known tolerance issues
- training producing checkpoints under `checkpoints/`
- evaluation producing `results/full_experiment.csv`
- plotting producing PNGs under `results/plots/`
- `run_trained.py` successfully loading a checkpoint and running episodes

## Important Implementation Notes

- The project uses a local copy of `gym-pybullet-drones`
- The active VS Code interpreter must point to the environment that has all dependencies installed
- `run_trained.py` has compatibility logic for older checkpoints whose policy layer names differ from the current network naming scheme
- The project currently uses separate scripts for training, evaluation, plotting, and visualization to keep concerns isolated

## Mental Model For The Whole System

A simple way to understand the project is this:

- The environment creates a surveillance problem
- The drones observe partial information
- They share intruder estimates through a noisy channel
- Trust scores decide whose messages matter
- MAPPO learns how to move based on those observations
- Evaluation checks whether trust improves resilience under attack

That is the full loop.

## Final Summary

SentryNet is not just a drone control project.
It is a controlled experimental framework for testing whether lightweight trust-aware communication can improve cooperative multi-agent reinforcement learning under adversarial conditions.

The key scientific contribution is the comparison between:

- baseline coordination with no trust
- coordination with packet loss only
- coordination with packet loss plus trust-aware message aggregation

If System C consistently outperforms System A under attack, the project supports the idea that decentralized trust estimation can improve robustness without cryptography.
