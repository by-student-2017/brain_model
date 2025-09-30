import numpy as np
from brain_region_base import BrainRegion

class PrefrontalCortex(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state=None):
        """
        Simulates processing in the prefrontal cortex based on neurotransmitter modulation.

        Parameters:
        - input_signal: list or array of input values representing sensory or cognitive stimuli.
        - neurotransmitters: dictionary containing neurotransmitter levels, e.g., {'dopamine': 1.0, 'glutamate': 1.0}
        - internal_state: optional parameter for future expansion (e.g., working memory state, stress level)

        Returns:
        - normalized output signal after neurotransmitter modulation.

        Scientific Notes:
        -----------------
        - Dopamine plays a critical role in prefrontal cortex function, particularly in cognitive tasks such as
          working memory, planning, decision-making, and response inhibition.
        - The effect of dopamine follows an inverse-U shaped relationship: both insufficient and excessive dopamine
          levels impair cognitive performance. Optimal dopamine levels enhance signal-to-noise ratio and task efficiency.
        - D1 dopamine receptors are especially important for working memory and are densely expressed in the lateral
          prefrontal cortex. Activation of D1 receptors improves neural responsiveness to relevant stimuli.
        - Glutamate is the primary excitatory neurotransmitter and contributes to cortical activation and synaptic plasticity.
        - Stress and aging can alter dopamine levels in the prefrontal cortex, leading to cognitive decline.
        - Genetic polymorphisms (e.g., COMT Val/Met variants) affect dopamine metabolism and influence individual
          differences in cognitive performance and drug response.
        - Traditional practices such as zazen and meditation may help regulate dopamine levels in the prefrontal cortex,
          potentially supporting self-control and cognitive stability. This is supported by primate studies showing that
          lateral prefrontal neurons are actively involved in self-control tasks during trained behavioral paradigms.
        - Classification rule switching and response inhibition are associated with specific subregions of the prefrontal cortex:
          the left lateral posterior ventral area is critical for switching classification rules, while the frontopolar cortex
          (prefrontal pole) is important for suppressing responses based on outdated rules. Primate studies show that neurons
          in these regions encode the current rule and evaluate feedback (correct vs incorrect), supporting flexible behavior.
        - The orbitofrontal cortex (BA11) is involved in value-based decision-making and reward evaluation. It modulates
          the relative weighting of expected outcomes and adjusts behavioral strategies accordingly. Dysfunction in BA11
          may lead to fixed reward sensitivity, impairing adaptive learning and reverse updating of value representations.

        Future Extensions:
        ------------------
        - Include dynamic dopamine regulation based on stress or task difficulty.
        - Model D1 receptor-specific modulation and inverse-U shaped performance curves.
        - Simulate genetic variability (e.g., COMT polymorphisms) and pharmacological interventions.
        - Implement rule switching and response inhibition mechanisms based on lateral and frontopolar prefrontal subregions.
        - Add orbitofrontal (BA11) module to simulate value ratio adjustment and feedback-based learning flexibility.
        """

        """
        前頭前野（Prefrontal Cortex）の処理を模擬するメソッド。

        パラメータ:
        - input_signal: 感覚刺激や認知的入力を表す数値配列。
        - neurotransmitters: 神経伝達物質の濃度を含む辞書（例：{'dopamine': 1.0, 'glutamate': 1.0}）。
        - internal_state: 将来的な拡張用（作業記憶、ストレスレベルなど）。

        戻り値:
        - 神経伝達物質によって調整された正規化出力信号。

        科学的背景:
        ----------------------------
        - 前頭前野は、作業記憶、計画、意思決定、反応抑制などの実行機能を担う。
        - ドーパミンは前頭前野の機能に重要で、特にD1受容体が作業記憶に関与。
        - ドーパミンの効果は逆U字型で、少なすぎても多すぎても認知機能が低下する。
        - グルタミン酸は興奮性の神経伝達物質で、皮質の活性化と可塑性に寄与。
        - ストレスや加齢はドーパミンの効率を低下させ、認知機能の衰えを引き起こす。
        - COMT遺伝子の多型（Val/Met）はドーパミン代謝に影響し、個人差を生む。
        - 座禅や瞑想はドーパミン調整に寄与し、自己制御や認知安定性を高める可能性がある。
        - サルの研究では、訓練された行動課題中に前頭前野の神経活動が自己制御に関与していることが示されている。
        - 前頭前野のサブ領域には以下のような機能分化がある：
            - 左側後部腹側領域：分類ルールの切り替え
            - 前頭極（BA10）：古いルールに基づく反応の抑制
            - 眼窩前頭皮質（BA11）：報酬評価と価値比の調整

        将来的な拡張ポイント:
        ----------------------------

        1. **動的なドーパミン調整**：
           - ストレスや課題難易度に応じてドーパミン濃度を変化させる。
           - 例：
             ```python
             if internal_state.get('stress', 0) > 0.5:
                 dopamine *= 0.8
             ```

        2. **D1受容体特異的なモジュレーション**：
           - D1受容体の活性化による作業記憶強化をモデル化。
           - 飽和や脱感作の動態も導入可能。

        3. **逆U字型の性能曲線**：
           - ドーパミン濃度と認知性能の非線形関係を再現。
           - 例：
             ```python
             def inverse_u(dopamine):
                 return -((dopamine - optimal_level)**2) + max_performance
             ```

        4. **遺伝的個体差のシミュレーション**：
           - COMT遺伝型によるドーパミン代謝の違いをモデル化。
           - 認知能力や薬物反応の個人差を再現。

        5. **ルール切り替えと反応抑制の実装**：
           - 左側前頭前野と前頭極の機能を分離し、柔軟な行動制御を実装。
           - フィードバック評価による行動更新も可能。

        6. **眼窩前頭皮質（BA11）の統合**：
           - 価値比の調整と報酬学習の柔軟性をモデル化。
           - 例：
             ```python
             value_ratio = expected_reward / perceived_cost
             ```

        7. **作業記憶バッファの追加**：
           - internal_state['working_memory'] を導入し、複数ステップの推論や目標保持を可能に。

        8. **瞑想・神経フィードバックの影響**：
           - 行動訓練による神経伝達物質の長期的調整をシミュレーション。

        9. **他領域との接続**：
           - 扁桃体（情動）、海馬（記憶）、島皮質（共感）、線条体（行動選択）との連携。

        10. **実験データとの整合性**：
            - fMRI、EEG、行動課題データとの対応を考慮したモデル設計。

        実装メモ：
        ----------------------------
        - 現在の実装では、ドーパミンとグルタミン酸による入力信号の重み付けを行う。
        - 指数関数変換により非線形な神経活性を模擬。
        - 正規化処理により競合的な符号化と確率的な意思決定出力を再現。
        """

        # Retrieve neurotransmitter levels with default values
        dopamine = neurotransmitters.get('dopamine', 1.0)
        glutamate = neurotransmitters.get('glutamate', 1.0)

        # Modulate input signal based on neurotransmitter levels
        # Dopamine and glutamate jointly influence signal strength and cortical responsiveness
        weighted_input = np.array(input_signal) * dopamine * glutamate

        # Apply exponential transformation to simulate nonlinear neural activation
        exp_input = np.exp(weighted_input)

        # Normalize to simulate competitive encoding and probabilistic output
        return exp_input / np.sum(exp_input)