# Method

## The model

The driver is not a network designed for driving. It is the measured wiring of a male *Drosophila* central nervous system from the MaleCNS release, run as a recurrent rate model:

- **Graph.** Every traced neuron is a unit with a signed rate state; every measured neuron-to-neuron connection is a directed edge. 165,122 neurons, 25,563,197 edges. Which neuron talks to which is fixed by the microscope and never changes.
- **What learns.** One gain per edge and one leak per neuron: 25,728,319 parameters, all constrained to the fly's wiring. Training never adds, removes or rewires a connection. This is the flyhard recipe.
- **State carried.** Four graph updates run per 50 ms decision, and the state is carried into the next decision. flyhard zeroes the state before every decision, which its author flags as a convenience for the wheel task, not a claim about biology; a real fly never resets. On the first driving task this one change took the same graph from 3 of 20 roads to 17 of 20 (`tasks/flat-road`, `results/flat-road`).
- **Sensory interface, frozen.** Each of the 4,114 optic-lobe sensory neurons reads one random pixel of the 64 × 32 frame with a random sign. Pixels are normalised to (value − 0.5) × 2 and clamped to ±2. Drawn once, never trained.
- **Motor interface, frozen.** A fixed random weight vector over the 708 motor neurons of the ventral nerve cord gives the steering angle through 0.5 rad × tanh(·). In the crossers task a second fixed vector gives target speed through 14 m/s × sigmoid(·).

Nothing engineered can carry the signal on its own: the only path from the pixels to the wheel runs through the measured graph.

## The task

`tasks/street/street_env.py`. A kinematic bicycle car (2.6 m wheelbase) at a fixed 8 m/s, steering-only, wheel slew limited to 4 rad/s and ±0.5 rad. A two-lane street (3.5 m lanes) on a sinusoidal centreline with buildings, lamp posts, parked cars intruding into the lane, oncoming cars at 8 m/s and a slow car at 3 m/s in the driver's lane that must be overtaken through the oncoming lane. The brain receives a ray-cast 64 × 32 grayscale frame every 50 ms. A street counts as completed only when all 500 decisions (200 m) are free of collisions and off-road events; failures are typed (off road, parked car, oncoming car, slow car, crosser).

Task versions, selected with `FLY_TASK_VERSION`: 3 = parked and oncoming only; 4 = slow car drawn grey with a 22 m pull-out; 5 = brighter slow car, 16 m pull-out (the reported traffic results); 6 = speed control, pedestrians and dogs.

## Training

Pure imitation, no reward.

1. **Demonstrations.** A pure-pursuit expert that reads the true state drives the 60 training streets (seeds 1000–1059) for 320 ticks each with noise injected into its executed steering, so the recordings include recovering from being off line. Overtake ticks are weighted 3× in every batch.
2. **Behaviour cloning.** Truncated backpropagation through time over windows of 16 decisions, batch 4 streets, Adam at learning rate 0.04, mean-squared error on the steering command normalised by the 0.5 rad range. 1,200 updates. This alone gave 12 of 20 held-out streets and a characteristic failure: steering of 1–2° where the expert commanded 8–18°, drifting into oncoming traffic after every overtake.
3. **DAgger.** Three rounds of 500 updates. The brain drives the training streets on its own, the expert relabels every state the brain actually reached, and training continues on the union. This closed the drift.
4. **Evaluation.** Three independent held-out sets of 20 streets (seeds 9000–9019, 9100–9119, 9200–9219), closed loop, with and without traffic, plus the mean lane offset compared with the expert's on the same streets.

## Controls

- **Linear map** (2,048 pixels → steering, 1,153 parameters) and **MLP-40** (one hidden layer, 46k parameters), trained on the same demonstrations; neither has memory.
- **Rewired fly.** The presynaptic index is permuted over all edges, which keeps every neuron's in-degree and the total edge count and destroys only which neuron connects to which. Trained with the identical recipe. It is the one comparison that asks whether the fly's specific wiring matters.
- **Seeds.** One training seed per condition (a second seed of the crossers model scored 13 of 20). Twenty streets per set is thin; we report gaps, not verdicts.

## The videos

Every drive is recorded once as per-tick state (`tasks/street/record_episode.py`): the car's pose and commands, every traffic and parked car, every crosser, the exact 64 × 32 frame the brain consumed, and the rate state of all neurons after each decision (int8, per-neuron scaled). `render/` rebuilds the scene from that log in Blender with Kenney's CC0 kits, so the look can change without changing what the brain saw or did. The panel on the right projects each neuron's measured soma position and colours it by its own recorded activity: orange positive, blue negative, grey quiet, each neuron scaled on its own range over the drive.
