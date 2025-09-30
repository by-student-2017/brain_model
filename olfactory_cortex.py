from brain_region_base import BrainRegion
import numpy as np

class OlfactoryCortex(BrainRegion):
    def __init__(self, name):
        super().__init__(name)
        self.previous_intensity = 0.0
        self.desensitization_rate = 0.1  # Sensitivity reduction rate

    def process(self, olfactory_input, neurotransmitters, internal_state):
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
