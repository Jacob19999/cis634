import pandas as pd

from asrsbench.labeling import import_labels
from asrsbench.tracks import read_qrels_tsv


def test_import_labels_computes_agreement_and_conservative_qrels(tmp_path):
    a = pd.DataFrame({"qid": ["T01", "T01", "T02"], "acn": ["1", "2", "3"], "grade": [2, 1, 0]})
    b = pd.DataFrame({"qid": ["T01", "T01"], "acn": ["1", "2"], "grade": [2, 0]})
    a.to_csv(tmp_path / "pool_A.csv", index=False)
    b.to_csv(tmp_path / "pool_B.csv", index=False)
    stats = import_labels([tmp_path / "pool_A.csv", tmp_path / "pool_B.csv"], tmp_path / "q.tsv")
    assert stats["n_double_labeled"] == 2
    assert stats["raw_agreement"] == 0.5
    # disagreement on (T01, 2) resolves to the lower grade
    assert read_qrels_tsv(tmp_path / "q.tsv") == {"T01": {"1": 2, "2": 0}, "T02": {"3": 0}}
