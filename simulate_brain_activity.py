# simulate_brain_activity_fixed.py

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
from olfactory_cortex import OlfactoryCortex  # Olfactory processing
from insula import Insula

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

neurotransmitters = config["neurotransmitters"]
internal_state = config["internal_state"]
external_stimuli = config["external_stimuli"]
image_signals = config["image_signals"]
linguistic_inputs = config["linguistic_inputs"]
auditory_inputs = config["auditory_inputs"]
olfactory_inputs = config["olfactory_inputs"]

def simulate_brain_activity(input_signal, neurotransmitters, external_stimuli, internal_state,
                            image_signals=None, linguistic_inputs=None, auditory_inputs=None, olfactory_inputs=None,
                            dt=0.1, steps=10, discrepancy_threshold=0.5, escape_duration=3):
    """
    脳活動のシミュレーションを行う統合関数。

    将来的な拡張ポイント：
    ============================

    1. 入力信号の多様化とモダリティ統合：
       - image_signals, linguistic_inputs, auditory_inputs, olfactory_inputs に加えて、
         触覚、味覚、内臓感覚などの入力モジュールを追加可能。
       - 各モダリティを統合する「感覚統合モジュール（Sensory Integration）」の導入。

    2. 内部状態の拡張：
       - internal_state に「睡眠状態」「疲労度」「ホルモン濃度」「免疫反応」などを追加。
       - 情動状態や報酬履歴を保持することで、学習と行動選択に反映可能。

    3. 社会的文脈の導入：
       - 他者の信頼度、関係性、集団所属などを social_context として扱い、
         Insula や Amygdala の処理に反映。
       - 共感、羞恥、罪悪感などの高次情動の生成に活用。

    4. 脳領域間の接続構造の明示化：
       - 各 BrainRegion を接続グラフとして定義し、信号の流れを動的に制御。
       - 例：海馬 → 前頭前野 → 線条体 → 運動出力 などの経路を構築。

    5. 学習機構の統合：
       - KalmanFilter に加えて、強化学習（Q学習、TD誤差）、ベイズ推定、Hebbian学習などを導入。
       - 報酬予測誤差に基づく神経伝達物質の動的調整。

    6. 記憶検索と再構成：
       - Hippocampus に MemorySearchEngine を追加し、過去の痕跡との照合と再構成を実装。
       - 類似記憶の再活性化による予測と行動選択の強化。

    7. 情動処理の高度化：
       - Amygdala に高次情動（尊敬、感謝、羞恥、罪悪感など）を追加。
       - Insula に社会的共感の時系列追跡機能を追加。

    8. 行動選択と運動出力の統合：
       - 線条体、前頭前野、視床、運動皮質などを連携させ、意思決定から運動出力までをモデル化。
       - escape_counter による逃避行動を、より複雑な行動選択に拡張。

    9. ホムンクルスフィードバックの視覚化：
       - homunculus_feedback を用いて、脳領域ごとの活動マップや情動マップを描画。
       - 時系列での情動変化や認知的不一致の推移を可視化。

    10. 実験データとの整合性：
        - fMRI、EEG、行動課題データとの対応を考慮したパラメータ設計。
        - 実験条件に応じた入力・内部状態の切り替え機能を追加。

    この関数は、脳モデルの統合的なシミュレーション基盤として設計されており、
    各モジュールの拡張と連携によって、より高度な認知・情動・行動モデルへと発展可能である。
    """

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
        AuditoryCortex("Auditory Cortex"),
        OlfactoryCortex("Olfactory Cortex"),
        Insula("Insula")
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
    olfactory_discomfort = []
    feedback_intensity = []
    emotion_states = []

    escape_counter = 0

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
        olfactory_input = olfactory_inputs[t % len(olfactory_inputs)] if olfactory_inputs else [0.0]

        outputs = {}
        for region in regions:
            if region.name == "Visual Cortex":
                outputs[region.name] = region.process(image_input, neurotransmitters, internal_state)
            elif region.name == "Language Area":
                outputs[region.name] = region.process(linguistic_input, neurotransmitters, internal_state)
            elif region.name == "Auditory Cortex":
                outputs[region.name] = region.process(auditory_input, neurotransmitters, internal_state)
            elif region.name == "Olfactory Cortex":
                outputs[region.name] = region.process(olfactory_input, neurotransmitters, internal_state)
            elif region.name == "Insula":
                observed_pain_signal = {"pain": 0.6, "distress": 0.4}  # 仮の入力
                outputs[region.name] = region.process(observed_pain_signal, internal_state)
            else:
                outputs[region.name] = region.process(input_signal, neurotransmitters, internal_state)

        observed_pain_signal = {"pain": 0.6, "distress": 0.4} # 仮の入力
        insula_output = outputs["Insula"] = regions[-1].process(observed_pain_signal, internal_state)

        visual_vs_language = float(np.linalg.norm(np.array(outputs["Visual Cortex"]) - np.array(outputs["Language Area"])))
        auditory_vs_language = float(np.linalg.norm(np.array(outputs["Auditory Cortex"]) - np.array(outputs["Language Area"])))
        olfactory_response = outputs["Olfactory Cortex"]
        if isinstance(olfactory_response, list):
            olfactory_response = olfactory_response[0]

        prefrontal_output = outputs["Prefrontal Cortex"]
        if isinstance(prefrontal_output, (list, np.ndarray)):
            prefrontal_output = float(prefrontal_output[0])
        else:
            prefrontal_output = float(prefrontal_output)

        adaptive_threshold = discrepancy_threshold \
            + 0.2 * prefrontal_output \
            + 0.3 * (neurotransmitters["serotonin"] - 0.5) \
            - 0.1 * np.exp(-escape_counter)

        discrepancy_value = max(visual_vs_language, auditory_vs_language)
        if discrepancy_value > adaptive_threshold:
            escape_counter += 1
        else:
            escape_counter = 0

        if escape_counter >= escape_duration:
            print(f"Step {t}: Due to a large gap in perception, escape mode is activated.")
            input_signal = np.zeros_like(input_signal)
            neurotransmitters['dopamine'] *= 0.5
            escape_counter = 0
            continue

        if olfactory_response > 0.7:
            print(f"Step {t}: Moldy smell detected. Escape mode triggered.")
            input_signal = np.zeros_like(input_signal)
            neurotransmitters['serotonin'] *= 0.7
            continue

        for item, info in consumption_history.items():
            if time - info["time"] < info["digest_time"]:
                print(f"Step {t}: {item} is excluded because it previously caused {info['symptom']}")
                neurotransmitters['serotonin'] *= 0.8

        reward = outputs["Striatum"][0] if isinstance(outputs["Striatum"], np.ndarray) else outputs["Striatum"]
        updated_dopamine = kf.update(reward)
        neurotransmitters['dopamine'] = updated_dopamine

        input_signal = outputs["Hippocampus"] * dt + input_signal * (1 - dt)

        emotion_state = {
            "fear": outputs["Amygdala"],
            "pleasure": neurotransmitters["dopamine"],
            "disgust": olfactory_response,
            "anger": max(0.0, visual_vs_language - 0.5),
            "empathy": insula_output
        }

        time_series.append(time)
        visual_language_discrepancy.append(visual_vs_language)
        auditory_language_discrepancy.append(auditory_vs_language)
        olfactory_discomfort.append(olfactory_response)
        feedback_intensity.append(discrepancy_value)
        emotion_states.append(emotion_state)

    homunculus_feedback = {
        "time": time_series,
        "visual_language_discrepancy": visual_language_discrepancy,
        "auditory_language_discrepancy": auditory_language_discrepancy,
        "olfactory_discomfort": olfactory_discomfort,
        "feedback_intensity": feedback_intensity,
        "emotion_states": emotion_states
    }

    with open("homunculus_feedback.json", "w") as f:
        json.dump(homunculus_feedback, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(time_series, visual_language_discrepancy, label="Visual-Language Discrepancy", color='blue')
    plt.plot(time_series, auditory_language_discrepancy, label="Auditory-Language Discrepancy", color='red')
    plt.plot(time_series, olfactory_discomfort, label="Olfactory Discomfort", color='green')
    plt.xlabel("Time (s)")
    plt.ylabel("Discrepancy / Discomfort")
    plt.title("Cognitive Discrepancies and Olfactory Discomfort Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("cognitive_discrepancies.png")
    plt.close()

# Example usage
initial_input = [0.1, 0.2, 0.3]
simulate_brain_activity(initial_input, neurotransmitters, external_stimuli, internal_state,
                        image_signals, linguistic_inputs, auditory_inputs, olfactory_inputs, steps=10)