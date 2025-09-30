# insula.py

"""
Insula class models the neural mechanisms of empathy-based altruistic behavior.
It processes observed pain signals from others, integrates internal bodily states,
and modulates emotional output toward the amygdala. This class is designed for
future extensibility, including reinforcement learning, social context integration,
and time-series tracking of empathy responses.

Key Features:
- Empathy signal processing based on observed pain
- Internal state integration (e.g., own pain, stress)
- Modulation of empathy strength and output to amygdala
- Short-term and long-term empathy memory
- Extensible architecture for reinforcement learning and social context modeling
"""

import numpy as np
from brain_region_base import BrainRegion

class Insula(BrainRegion):
    """
    Insula class models the neural mechanisms of empathy-based altruistic behavior.
    
    概要：
    ----------------------------
    - 島皮質（Insula）は、自己の身体状態と他者の感情状態を統合することで、
      共感に基づく利他的行動を促進する脳領域。
    - 本クラスは、観察された痛み信号（他者の苦痛）と自己の内部状態（例：自身の痛み、ストレス）を統合し、
      扁桃体への情動出力として共感強度を生成する。
    - 短期記憶（STP）と長期記憶（LTP）を用いて、共感反応の学習と蓄積を可能にする。
    - 将来的な拡張として、強化学習、社会的文脈の統合、時間的変化の追跡などに対応可能。
    
    主な機能：
    ----------------------------
    - 観察された痛み信号に基づく共感処理
    - 内部状態（自身の痛み、ストレスなど）との統合
    - 共感強度の調整と扁桃体への出力
    - STP（短期共感記憶）とLTP（長期共感記憶）の更新
    - 拡張性の高いアーキテクチャ設計
    
    拡張ポイント：
    ============================
    
    1. 強化学習による共感重みの調整：
       - empathy_weights を報酬に基づいて動的に更新し、社会的報酬に応じた共感反応を学習。
       - 例：
         ```python
         reward = social_feedback.get('altruism_reward', 0.0)
         self.empathy_weights['pain'] += alpha * (reward - self.empathy_weights['pain'])
         ```
    
    2. 時系列共感応答の追跡：
       - 共感反応を時間軸で記録し、共感の持続性や変化をモデル化。
       - 例：
         ```python
         self.empathy_history.append((timestamp, empathy_output))
         ```
    
    3. 社会的文脈の統合：
       - social_context に「関係性」「集団所属」「親密度」などを含め、共感強度を調整。
       - 例：
         ```python
         familiarity = social_context.get('familiarity', 1.0)
         empathy_signal *= familiarity
         ```
    
    4. 扁桃体・前頭前野との接続：
       - 扁桃体への情動出力、前頭前野による行動制御との連携を設計。
       - 例：
         ```python
         amygdala_input = self.process(...)  # 情動強度
         prefrontal_modulation = prefrontal_cortex.decide_action(amygdala_input)
         ```
    
    5. 共感カテゴリの拡張：
       - pain, distress 以外にも「羞恥」「罪悪感」「感謝」などの高次共感カテゴリを追加。
       - 例：
         ```python
         self.empathy_weights['guilt'] = 1.0
         ```
    
    6. 実験データとの整合性：
       - fMRI、皮膚電気反応、表情認識などの生理・行動データと対応する共感強度の調整。
       - 例：
         ```python
         if internal_state.get('SCR') > threshold:
             empathy_output *= 1.2
         ```
    
    このクラスは、神経科学的に忠実な共感処理モデルとして、社会的・倫理的・文化的な文脈を含む
    高度な利他行動の生成に対応可能な設計となっており、今後の拡張に柔軟に対応できる。
    """

    def __init__(self, name="Insula"):
        super().__init__(name)
        self.empathy_weights = {
            'pain': 1.0,
            'distress': 1.0,
            'fear': 1.0,
            'sadness': 1.0
        }
        self.short_term_empathy = {}  # STP-like temporary empathy memory
        self.long_term_empathy = {}   # LTP-like persistent empathy memory
        self.stp_decay_rate = 0.9     # Short-term empathy decay rate
        self.ltp_learning_rate = 0.01 # Long-term empathy learning rate

    def update_empathy_memory(self, signal_type, signal_strength):
        # Short-term potentiation
        self.short_term_empathy[signal_type] = signal_strength

        # Long-term potentiation
        prev = self.long_term_empathy.get(signal_type, 1.0)
        self.long_term_empathy[signal_type] = prev + self.ltp_learning_rate * (signal_strength - prev)

    def decay_stp(self):
        for signal_type in self.short_term_empathy:
            self.short_term_empathy[signal_type] *= self.stp_decay_rate

    def process(self, observed_pain_signal, internal_state=None, social_context=None):
        """
        Process observed pain signal and internal state to generate empathy output.
        Parameters:
            observed_pain_signal (dict): e.g., {'pain': 0.8, 'distress': 0.6}
            internal_state (dict): e.g., {'own_pain': 0.5, 'stress': 0.3}
            social_context (dict): optional future extension
        Returns:
            empathy_output (float): strength of empathy signal to amygdala
        """
        if internal_state is None:
            internal_state = {}

        # Combine observed pain and internal state
        empathy_signal = 0.0
        for signal_type, strength in observed_pain_signal.items():
            internal_modulation = internal_state.get('own_' + signal_type, 1.0)
            weight = self.empathy_weights.get(signal_type, 1.0)
            combined_strength = strength * internal_modulation * weight
            empathy_signal += combined_strength

            # Update memory
            self.update_empathy_memory(signal_type, combined_strength)

        # Normalize output
        empathy_output = np.tanh(empathy_signal)

        return empathy_output

    # Future extension points:
    # - Incorporate reinforcement learning to adjust empathy_weights
    # - Use time-series tracking of empathy responses for adaptive behavior
    # - Integrate social context (e.g., group membership, familiarity)
    # - Connect to amygdala and prefrontal cortex for action modulation