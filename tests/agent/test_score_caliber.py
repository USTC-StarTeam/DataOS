"""Every score the leader sees carries the eval size that produced it.

The leader picks each experiment's eval size (AUTO mode), so a 25-item probe and the full
benchmark would otherwise both reach it as bare floats — and "did this beat its parent?" becomes
a comparison between two different measurements. A 4b2 batch was lost that way: bests landed on
5 / 8 / 25 / 128 / 200 / full, none of them comparable to the baseline. Annotating the number
keeps the judgement with the leader instead of hiding the problem.
"""

from pathlib import Path

from dataos.agent.state_tree import TreeState


def _tree(tmp_path: Path) -> TreeState:
    return TreeState.load(tmp_path, "base-model")


def _add_scored(tree: TreeState, *, score, eval_limit, name="e"):
    node_id = tree.add_planned_node(parent=tree.root_id, round_idx=0, plan={"name": name})
    tree.attach_report(node_id, report_path="/r.json", status="completed",
                       score=score, eval_limit=eval_limit, model_path="/out")
    return node_id


def test_subset_and_full_scores_are_distinguishable(tmp_path):
    _add_scored(_t := _tree(tmp_path), score=74.0, eval_limit=200, name="probe")
    _add_scored(_t, score=84.69, eval_limit=-1, name="confirm")
    scores = [n["score"] for n in _t.to_context()["nodes"] if n["score"]]
    assert "74.00@200" in scores
    assert "84.69@full" in scores


def test_subset_shows_what_fraction_of_the_benchmark_it_is(tmp_path):
    """With the benchmark's real size known, "@200/1319" says how thin the probe was."""
    _add_scored(_t := _tree(tmp_path), score=74.0, eval_limit=200, name="probe")
    _add_scored(_t, score=84.69, eval_limit=-1, name="confirm")
    scores = [n["score"] for n in _t.to_context(full_size=1319)["nodes"] if n["score"]]
    assert "74.00@200/1319" in scores
    assert "84.69@full" in scores  # a full eval needs no fraction


def test_unrecorded_caliber_is_marked_not_assumed(tmp_path):
    """A pre-AUTO report has no eval.limit — say so rather than implying it was full."""
    _add_scored(_t := _tree(tmp_path), score=37.8, eval_limit=None)
    assert [n["score"] for n in _t.to_context()["nodes"] if n["score"]] == ["37.80@?"]


def test_unscored_node_stays_none(tmp_path):
    assert all(n["score"] is None for n in _tree(tmp_path).to_context()["nodes"])


def test_tree_view_is_tiered(tmp_path):
    """The view is re-sent every round and grows with the tree, so only the newest nodes carry
    prose. Measured on a real 31-node run, hypothesis+review was 92% of a 40K view."""
    t = _tree(tmp_path)
    for i in range(t.PROSE_NODES + 5):
        nid = t.add_planned_node(parent=t.root_id, round_idx=i,
                                 plan={"name": f"e{i}", "hypothesis": "H" * 300})
        t.attach_report(nid, report_path="/r.json", status="completed", score=50.0 + i,
                        eval_limit=-1, model_path="/out")
    nodes = t.to_context()["nodes"]
    with_prose = [n for n in nodes if "hypothesis" in n]
    assert len(with_prose) == t.PROSE_NODES
    assert with_prose == nodes[-t.PROSE_NODES:]          # the newest ones
    for n in nodes[:-t.PROSE_NODES]:                     # older: structure only
        assert set(n) == {"node_id", "parent_node_id", "depth", "round", "status",
                          "experiment_name", "has_model", "score"}


def test_tree_view_never_repeats_the_reviews(tmp_path):
    """recent_reviews carries them whole; duplicating lesson/recommendation here was 23K/round."""
    t = _tree(tmp_path)
    nid = t.add_planned_node(parent=t.root_id, round_idx=0, plan={"name": "e"})
    t.attach_report(nid, report_path="/r.json", status="completed", score=1.0,
                    eval_limit=-1, model_path="/m")
    t.attach_review(nid, review_path="/v.json",
                    review={"lesson": "L", "recommendation_for_future_growth": "R"})
    assert all("review" not in n for n in t.to_context()["nodes"])


def test_the_node_keeps_the_prediction_the_review_needs(tmp_path):
    """expected_signal was written every round and read by nobody: the leader got no feedback on
    its own judgement. The node keeps it, and the review context now lifts it out of node.plan."""
    t = _tree(tmp_path)
    nid = t.add_planned_node(parent=t.root_id, round_idx=1, plan={
        "name": "e", "hypothesis": "H", "expected_signal": "+3 points over the parent"})
    node = t.get(nid)
    assert node["hypothesis"] == "H"
    assert node["plan"]["expected_signal"] == "+3 points over the parent"


def test_review_prompt_asks_for_the_verdict():
    from dataos import prompt

    text = prompt.render("leader/turn/review", context_json="{}")
    assert "prediction" in text and "expected_signal" in text
