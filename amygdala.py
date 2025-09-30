import numpy as np
from brain_region_base import BrainRegion

class Amygdala(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state=None):
        """
        Amygdala class models the emotional processing center of the brain.
        
        概要：
        ----------------------------
        - 扁桃体は情動の評価・記憶・反応生成に関与する脳領域であり、特に恐怖・信頼・怒りなどの
          情動反応を統合する役割を持つ。
        - 本クラスは、入力信号と神経伝達物質（ノルアドレナリン、セロトニン）を用いて
          情動強度を計算し、信頼スコアに基づいて情動ラベルを選定する。
        - 短期増強（STP）と長期増強（LTP）をサポートし、情動記憶の動的な更新が可能。
        - 信頼スコアは recent_response に基づいて更新され、社会的文脈に応じた情動反応を生成。
        
        構造：
        ----------------------------
        - emotional_weights: 各情動カテゴリに対する重み（信頼度など）
        - short_term_memory: STP用の一時的な情動強度記録
        - long_term_memory: LTP用の恒常的な情動強度記録
        - stp_decay_rate: STPの減衰率（例：毎ステップで10%減衰）
        - ltp_learning_rate: LTPの学習率（例：新しい刺激に対する重みの更新速度）
        
        処理の流れ：
        ----------------------------
        1. 入力信号の最大値を取得し、ノルアドレナリンとセロトニンで調整。
        2. internal_state から信頼スコアと recent_response を取得。
        3. recent_response に応じて信頼スコアを更新。
        4. 信頼スコアに基づいて selected_emotion を選定（例：信頼 vs 恐怖）。
        5. emotional_weights を用いて weighted_emotion を計算。
        6. 信頼スコアを internal_state に保存して返却。
        
        拡張ポイント：
        ============================
        
        1. 信頼スコアの時系列学習：
           - recent_response を履歴として蓄積し、時系列モデル（例：指数移動平均）で信頼度を更新。
           - 例：
             ```python
             trust_score = ema(recent_responses)
             ```
        
        2. 情動分類器の導入：
           - selected_emotion を入力特徴量（例：音声、表情、文脈）から分類器で推定。
           - 例：
             ```python
             selected_emotion = emotion_classifier.predict(input_features)
             ```
        
        3. 強化学習との統合：
           - trust_score を報酬学習で更新し、社会的報酬に基づく信頼形成をモデル化。
           - 例：
             ```python
             trust_score += alpha * (reward - trust_score)
             ```
        
        4. 高次情動の拡張：
           - 尊敬、羞恥、罪悪感、感謝などの高次情動に対してもスコアを導入し、文脈依存の反応を生成。
           - 例：
             ```python
             self.emotional_weights['guilt'] = guilt_score
             ```
        
        5. 他領域との連携：
           - 前頭前野（意思決定）、海馬（記憶）、視床（感覚統合）などと連携し、情動反応の統合処理を実装。
           - 例：
             ```python
             if hippocampus.memory_match(input_signal):
                 selected_emotion = 'surprise'
             ```
        
        6. 実験データとの整合性：
           - fMRIや皮膚電気反応（SCR）などの生理指標と対応する情動強度の調整。
           - 例：
             ```python
             if internal_state.get('SCR') > threshold:
                 signal_strength *= 1.2
             ```
        
        このクラスは、情動処理の神経科学的モデルとして、社会的文脈・記憶・信頼・報酬を統合する
        高度な情動反応生成を可能にする設計となっており、今後の拡張に柔軟に対応できる。
        """
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
