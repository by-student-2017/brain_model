import numpy as np
from brain_region_base import BrainRegion

class Amygdala(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state=None):
        norepinephrine = neurotransmitters.get('norepinephrine', 1.0)
        serotonin = neurotransmitters.get('serotonin', 1.0)
        signal_strength = np.max(input_signal) * norepinephrine * serotonin
        return signal_strength if signal_strength > 0.3 else 0

'''
"""
Amygdala class models the emotional processing center of the brain.
It receives sensory input and internal state signals, applies neurotransmitter modulation,
and outputs weighted emotional responses. It supports short-term potentiation (STP),
long-term potentiation (LTP), and dynamic trust-based emotional weighting.
This class is designed for extensibility, allowing future integration with classifiers,
reinforcement learning, and expanded emotion modeling.
"""
import numpy as np
from brain_region_base import BrainRegion

class Amygdala(BrainRegion):
    def __init__(self):
        super().__init__()
        self.emotional_weights = {...}  # 既存の情動カテゴリ
        self.short_term_memory = {}     # STP用の一時的な重み
        self.long_term_memory = {}      # LTP用の恒常的な重み
        self.stp_decay_rate = 0.9       # 短期記憶の減衰率（例：毎ステップで10%減衰）
        self.ltp_learning_rate = 0.01   # 長期記憶の学習率

    def update_memory(self, emotion, signal_strength):
        # 短期増強（STP）
        self.short_term_memory[emotion] = signal_strength

        # 長期増強（LTP）
        prev = self.long_term_memory.get(emotion, 1.0)
        self.long_term_memory[emotion] = prev + self.ltp_learning_rate * (signal_strength - prev)

    def decay_stp(self):
        for emotion in self.short_term_memory:
            self.short_term_memory[emotion] *= self.stp_decay_rate
        
        # 情動カテゴリとそれぞれの重み（初期値）
        self.emotional_weights = {
            # 基本情動
            'fear': 1.0,
            'pleasure': 1.0,
            'disgust': 1.0,
            'anger': 1.0,
            'surprise': 1.0,
            'sadness': 1.0,
            
            # 高次情動（社会的・倫理的・文化的）
            'curiosity': 1.0,
            'trust': 1.0,
            'anticipation': 1.0,
            'pride': 1.0,
            'shame': 1.0,
            'guilt': 1.0,
            'gratitude': 1.0,
            'respect': 1.0,
            'envy': 1.0,
            'humility': 1.0,
            'honor': 1.0,
            'embarrassment': 1.0,
            'compassion': 1.0,
        }

    def process(self, input_signal, neurotransmitters, internal_state=None):
        norepinephrine = neurotransmitters.get('norepinephrine', 1.0)
        serotonin = neurotransmitters.get('serotonin', 1.0)
        signal_strength = np.max(input_signal) * norepinephrine * serotonin

        # --- 信頼度の構築と更新 ---
        if internal_state is None:
            internal_state = {}

        # 初期信頼度の取得
        trust_score = internal_state.get('trust_score', 1.0)

        # 過去の反応履歴に基づいて信頼度を更新（例：ポジティブ反応なら増加）
        recent_response = internal_state.get('recent_response', 'neutral')  # 'positive', 'negative', 'neutral'
        if recent_response == 'positive':
            trust_score = min(trust_score + 0.05, 1.0)
        elif recent_response == 'negative':
            trust_score = max(trust_score - 0.1, 0.0)
        # neutral の場合は変化なし

        # 更新された信頼度を emotional_weights に反映
        self.emotional_weights['trust'] = trust_score

        # --- 情動ラベルの選定（仮） ---
        selected_emotion = 'trust' if trust_score > 0.5 and signal_strength > 0.3 else 'fear'

        weighted_emotion = signal_strength * self.emotional_weights.get(selected_emotion, 1.0)

        # --- internal_state に更新された trust_score を保存 ---
        internal_state['trust_score'] = trust_score

        return weighted_emotion if weighted_emotion > 0.3 else 0

        # --- 拡張ポイント ---
        # - recent_response を複数履歴で蓄積し、時系列で信頼度を学習
        # - selected_emotion を分類器で推定（例：入力特徴量から）
        # - trust_score を報酬学習や強化学習で更新
        # - 他の情動（例：尊敬、羞恥）にも同様のスコアを導入可能
'''