from brain_region_base import BrainRegion
import numpy as np

class OlfactoryCortex(BrainRegion):
    def __init__(self, name):
        super().__init__(name)
        self.previous_intensity = 0.0
        self.desensitization_rate = 0.1  # Sensitivity reduction rate

    def process(self, olfactory_input, neurotransmitters, internal_state):
        """
        嗅覚入力に基づいて不快感を計算するメソッド。
        
        現在の実装：
        ----------------------------
        - olfactory_input の強度を取得し、前回との変化量（delta）を計算。
        - 刺激が継続している場合は desensitization_rate に基づいて感度を低下。
        - セロトニン濃度が高いほど不快感が抑制されるように調整。
        
        将来的な拡張ポイント（具体例付き）：
        ============================
        
        1. 嗅覚受容体の種類ごとの反応モデル：
           - OR（一般的な匂い）、TAAR（危険信号）、GC-D（CO₂やペプチド）、MS4A（フェロモン）など。
           - 例：
             ```python
             receptor_response = {
                 "OR": intensity * 1.0,
                 "TAAR": intensity * 1.5 if intensity > 0.7 else 0.0,
                 "GC-D": intensity * 0.8 if internal_state.get("CO2_level", 0) > 0.5 else 0.0
             }
             ```
        
        2. combinatorial coding による匂い識別：
           - 複数の受容体が同時に反応することで匂いの種類を識別。
           - 例：
             ```python
             odor_profile = np.array([OR_response, TAAR_response, GC_D_response])
             odor_id = np.argmax(odor_profile)  # 最も強く反応した受容体で匂いを分類
             ```
        
        3. 順応と脱感作の時間スケール調整：
           - 刺激の持続時間に応じて desensitization_rate を動的に変更。
           - 例：
             ```python
             exposure_duration = internal_state.get("odor_exposure_time", 0)
             self.desensitization_rate = min(0.5, 0.1 + 0.01 * exposure_duration)
             ```
        
        4. 遺伝的個体差や疾患モデルの導入：
           - 嗅覚受容体の遺伝子多型（例：OR2J3の変異）による感度差を反映。
           - がん細胞での嗅覚受容体発現など、病態モデルとの接続も可能。
        
        5. 行動誘導との連携：
           - 匂い刺激に応じて逃避・接近行動を選択。
           - 例：
             ```python
             if TAAR_response > 0.8:
                 internal_state["escape_mode"] = True
             ```
        
        6. 報酬学習との統合：
           - 好ましい匂い（例：食物）に対して接近行動を強化。
           - 嫌悪刺激（例：腐敗臭）に対して回避行動を学習。
           - 例：
             ```python
             reward_signal = 1.0 if odor_id == "food" else -1.0
             ```
        
        7. 海馬・扁桃体との連携による嗅覚記憶：
           - 匂いと記憶・情動を結びつけることで、懐かしさや警戒感を生成。
           - 例：
             ```python
             if odor_id in hippocampus.long_term_memory:
                 emotion = "nostalgia"
             ```
        
        8. 感情反応の複雑化：
           - 匂いの組み合わせによって複雑な情動（例：懐かしさ＋警戒）を生成。
           - 例：
             ```python
             if OR_response > 0.5 and TAAR_response > 0.5:
                 emotion = "mixed_feelings"
             ```
        
        これらの拡張により、嗅覚皮質は単なる匂い処理を超えて、
        情動・記憶・行動選択に関与する高度な認知モジュールとして機能可能になる。
        """
        intensity = np.array(olfactory_input)[0]
        delta = abs(intensity - self.previous_intensity)

        # If stimulation continues, gradually reduce sensitivity
        if delta < 0.05:
            intensity *= (1 - self.desensitization_rate)

        self.previous_intensity = intensity

        serotonin_level = neurotransmitters.get("serotonin", 0.5)
        discomfort = intensity * (1 - serotonin_level)

        # 将来的な拡張のためのコメント:
        # - 嗅覚受容体の種類（OR, TAAR, GC-D, MS4A）に応じた反応特性を追加可能
        # - combinatorial coding（多対多の受容体応答）に基づく匂い識別モデルを導入可能
        # - 順応（adaptation）や脱感作（desensitization）の時間スケールを動的に調整可能
        # - 嗅覚受容体の遺伝子多型や疾患関連（例：がん細胞での発現）を考慮した個体差モデル
        # - 匂い刺激による行動誘導（例：逃避、接近）を他モジュールと連携して実装可能
        
        # - TAAR（微量アミン受容体）による危険信号の検出モデルを追加可能
        # - GC-D受容体によるCO₂やペプチドの検出モデルを追加可能
        # - 嗅覚受容体の種類ごとの反応特性を辞書型で管理し、入力に応じて分岐処理を実装可能
        # - 匂い刺激に対する報酬学習（例：好ましい匂いへの接近、嫌悪刺激の回避）を強化学習モジュールと連携して実装可能
        # - 嗅覚記憶と報酬の関連づけを海馬や扁桃体モジュールと連携して構築可能
        # - 匂いの組み合わせによる複雑な感情反応（例：懐かしさ、警戒）をモデル化可能

        return [discomfort]
