'''
class BrainRegion:
    def __init__(self, name):
        self.name = name

    def process(self, input_signal, neurotransmitters, internal_state=None):
        raise NotImplementedError
'''
class BrainRegion:
    def __init__(self, name):
        self.name = name
        self.related_terms = []  # 関連項目（例：神経伝達物質、機能、疾患など）
        self.effects = []        # 関連する効果（例：抑制、興奮、記憶形成など）
        self.connections = {}    # 他の脳領域との接続情報（例：{'海馬': '双方向'}）

    def add_related_term(self, term):
        """関連項目を追加する"""
        self.related_terms.append(term)

    def add_effect(self, effect):
        """関連する効果を追加する"""
        self.effects.append(effect)

    def connect_to(self, other_region, connection_type="unknown"):
        """他の脳領域との接続を記録する"""
        self.connections[other_region] = connection_type

    def process(self, input_signal, neurotransmitters, internal_state=None):
        """
        入力信号を処理するメソッド。
        実装は各脳領域の特性に応じて将来的に拡張される予定。
        """
        raise NotImplementedError("このメソッドは将来的に各脳領域に応じて実装されます。")

    def __repr__(self):
        return f"<BrainRegion: {self.name}>"
