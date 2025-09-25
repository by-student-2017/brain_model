# brain_model

## This table summarizes the major brain regions, their primary functions, typical input and output information, and associated neurotransmitters. It serves as a foundation for constructing simplified mathematical models of brain function.

| Brain Region     | Primary Function         | Input Information                  | Output Information                  | Related Neurotransmitters               |
|------------------|--------------------------|------------------------------------|-------------------------------------|-----------------------------------------|
| Prefrontal Cortex| Decision-making, planning| External stimuli, memory, reward   | Action selection, judgment          | Dopamine, Glutamate                     |
| Striatum         | Reward processing        | Reward prediction error            | Dopamine release level              | Dopamine                                |
| Hippocampus      | Memory formation, spatial cognition | Sensory data, contextual information | Memory traces, recall signals     | Glutamate, Acetylcholine     |
| Amygdala         | Emotional response       | Fear and threat stimuli            | Emotional intensity, avoidance behavior | Norepinephrine, Serotonin           |
| Hypothalamus     | Hormonal regulation      | Internal state, environmental stimuli | Hormone release commands         | Oxytocin, Vasopressin                   |
| Cerebellum       | Motor control            | Motor commands, sensory feedback   | Fine motor adjustments              | GABA, Glutamate                         |
| Midbrain         | Attention, arousal       | Sensory stimuli, reward signals    | Focused attention, arousal level    | Acetylcholine, Dopamine                 |
| Brainstem        | Vital functions          | Breathing, heart rate signals      | Autonomic control signals           | Serotonin, Norepinephrine               |


project/
│
├── kalman_filter.py
├── brain_region_base.py
├── prefrontal_cortex.py
├── striatum.py
├── hippocampus.py
├── amygdala.py
├── hypothalamus.py
├── cerebellum.py
├── midbrain.py
├── brainstem.py
├── visual_cortex.py
├── language_area.py
├── auditory_cortex.py
│
└── simulate_brain_activity.py
