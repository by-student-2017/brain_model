class BrainRegion:
    def __init__(self, name):
        self.name = name
        self.related_terms = []  # 関連項目（例：神経伝達物質、機能、疾患など）
        self.effects = []        # 関連する効果（例：抑制、興奮、記憶形成、可塑性など）
        self.connections = {}    # 他の脳領域との接続情報（例：{'海馬': '双方向'}）

    def add_related_term(self, term):
        """関連項目を追加する。
        将来的には、脳科学辞典の用語（例：GABA、長期増強、ワーキングメモリー）とリンクさせ、
        機能的・構造的な意味づけを自動的に付与することが可能。
        """
        self.related_terms.append(term)

    def add_effect(self, effect):
        """関連する効果を追加する。
        例：抑制（GABA）、興奮（グルタミン酸）、記憶形成（海馬）、報酬評価（眼窩前頭皮質）など。
        将来的には、効果の強度や持続時間、神経伝達物質との関連性も記録可能。
        """
        self.effects.append(effect)

    def connect_to(self, other_region, connection_type="unknown"):
        """他の脳領域との接続を記録する。
        connection_type には '双方向', '片方向', '抑制性', '興奮性' などを指定可能。
        将来的には、接続強度やシナプス密度、可塑性の履歴も記録できるように拡張。
        """
        self.connections[other_region] = connection_type

    def process(self, input_signal, neurotransmitters, internal_state=None):
        """
        入力信号を処理するメソッド。
        実装は各脳領域の特性に応じて将来的に拡張される予定。

        拡張ポイント：
        ----------------------------
        - 各 BrainRegion サブクラスで、GABA（抑制）、グルタミン酸（興奮）、ドーパミン（報酬）などの
          神経伝達物質の影響をモデル化。
        - internal_state にストレス、疲労、覚醒度、ホルモン濃度などを含め、状態依存の処理を実装。
        - 脳科学辞典に記載されている疾患（例：統合失調症、うつ病、アルツハイマー病）との関連性を
          self.related_terms に記録し、病態モデルとの接続を可能に。
        - 長期増強（LTP）、短期増強（STP）、スパイクタイミング依存可塑性（STDP）などの
          可塑性メカニズムを self.effects に記録し、学習履歴に応じた動的処理を実装。
        - connect_to により、Papez回路、報酬系、感覚統合系などのネットワーク構造を再現可能。
        """
        raise NotImplementedError("このメソッドは将来的に各脳領域に応じて実装されます。")

    def __repr__(self):
        return f"<BrainRegion: {self.name}>"
