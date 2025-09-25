import numpy as np
import json
import matplotlib.pyplot as plt

# Import all brain region modules and KalmanFilter
from kalman_filter import KalmanFilter
from brain_region_base import BrainRegion
from prefrontal_cortex import PrefrontalCortex
from striatum import Striatum
from hippocampus import Hippocampus
from amygdala import Amygdala
from hypothalamus import Hypothalamus
from cerebellum import Cerebellum
from midbrain import Midbrain
from brainstem import Brainstem
from visual_cortex import VisualCortex
from language_area import LanguageArea
from auditory_cortex import AuditoryCortex

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

neurotransmitters = config["neurotransmitters"]
internal_state = config["internal_state"]
external_stimuli = config["external_stimuli"]
image_signals = config["image_signals"]
linguistic_inputs = config["linguistic_inputs"]
auditory_inputs = config["auditory_inputs"]

# === Simulation Function with Kalman Filter Integration and Escape Mode ===
def simulate_brain_activity(input_signal, neurotransmitters, external_stimuli, internal_state,
                            image_signals=None, linguistic_inputs=None, auditory_inputs=None,
                            dt=0.1, steps=10, discrepancy_threshold=0.5, escape_duration=3):
    regions = [
        PrefrontalCortex("Prefrontal Cortex"),
        Striatum("Striatum"),
        Hippocampus("Hippocampus"),
        Amygdala("Amygdala"),
        Hypothalamus("Hypothalamus"),
        Cerebellum("Cerebellum"),
        Midbrain("Midbrain"),
        Brainstem("Brainstem"),
        VisualCortex("Visual Cortex"),
        LanguageArea("Language Area"),
        AuditoryCortex("Auditory Cortex")
    ]

    kf = KalmanFilter(
        initial_state=neurotransmitters['dopamine'],
        initial_uncertainty=0.1,
        process_variance=0.01,
        observation_variance=0.05
    )

    time_series = []
    visual_language_discrepancy = []
    auditory_language_discrepancy = []
    feedback_intensity = []

    # Track discrepancy duration for escape mode
    escape_counter = 0

    # Sample consumption history for enhanced learning
    consumption_history = {
        "milk": {"time": 2.0, "digest_time": 4.0, "symptom": "abdominal pain", "organ": "stomach"},
        "shrimp": {"time": 5.0, "digest_time": 6.0, "symptom": "itch", "organ": "skin"}
    }

    for t in range(steps):
        time = t * dt
        stimulus = external_stimuli[t % len(external_stimuli)]
        input_signal = np.array(input_signal) + np.array(stimulus)

        image_input = image_signals[t % len(image_signals)] if image_signals else [0.5, 0.5, 0.5]
        linguistic_input = linguistic_inputs[t % len(linguistic_inputs)] if linguistic_inputs else [0.2, 0.3]
        auditory_input = auditory_inputs[t % len(auditory_inputs)] if auditory_inputs else [0.1, 0.4]

        outputs = {}
        for region in regions:
            if region.name == "Visual Cortex":
                outputs[region.name] = region.process(image_input, neurotransmitters, internal_state)
            elif region.name == "Language Area":
                outputs[region.name] = region.process(linguistic_input, neurotransmitters, internal_state)
            elif region.name == "Auditory Cortex":
                outputs[region.name] = region.process(auditory_input, neurotransmitters, internal_state)
            else:
                outputs[region.name] = region.process(input_signal, neurotransmitters, internal_state)

        visual_vs_language = abs(outputs["Visual Cortex"] - outputs["Language Area"])
        auditory_vs_language = abs(outputs["Auditory Cortex"] - outputs["Language Area"])

        # Check for escape condition
        if max(visual_vs_language, auditory_vs_language) > discrepancy_threshold:
            escape_counter += 1
        else:
            escape_counter = 0

        if escape_counter >= escape_duration:
            print(f"Step {t}: Due to a large gap in perception, escape mode is activated.")
            input_signal = np.zeros_like(input_signal)
            neurotransmitters['dopamine'] *= 0.5
            escape_counter = 0
            continue

        # Check consumption history for physiological effects
        for item, info in consumption_history.items():
            if time - info["time"] < info["digest_time"]:
                print(f"Step {t}: {item} is excluded because it previously caused {info['symptom']}")
                neurotransmitters['serotonin'] *= 0.8

        reward = outputs["Striatum"][0] if isinstance(outputs["Striatum"], np.ndarray) else outputs["Striatum"]
        updated_dopamine = kf.update(reward)
        neurotransmitters['dopamine'] = updated_dopamine

        input_signal = outputs["Hippocampus"] * dt + input_signal * (1 - dt)

        time_series.append(time)
        visual_language_discrepancy.append(visual_vs_language)
        auditory_language_discrepancy.append(auditory_vs_language)
        feedback_intensity.append(max(visual_vs_language, auditory_vs_language))

    # Save homunculus feedback data
    homunculus_feedback = {
        "time": time_series,
        "visual_language_discrepancy": visual_language_discrepancy,
        "auditory_language_discrepancy": auditory_language_discrepancy,
        "feedback_intensity": feedback_intensity
    }

    with open("homunculus_feedback.json", "w") as f:
        json.dump(homunculus_feedback, f, indent=2)

    # Plot cognitive discrepancies
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, visual_language_discrepancy, label="Visual-Language Discrepancy", color='blue')
    plt.plot(time_series, auditory_language_discrepancy, label="Auditory-Language Discrepancy", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Discrepancy")
    plt.title("Cognitive Discrepancies Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("cognitive_discrepancies.png")
    plt.close()

# Example usage
initial_input = [0.1, 0.2, 0.3]
simulate_brain_activity(initial_input, neurotransmitters, external_stimuli, internal_state,
                        image_signals, linguistic_inputs, auditory_inputs, steps=10)
