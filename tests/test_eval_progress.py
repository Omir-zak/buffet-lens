from eval import progress
from eval.progress import Progress


def test_results_survive_a_restart_and_completed_items_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_DIR", tmp_path)
    first = Progress("answers")
    first.put("q1", {"ok": True})
    first.put("q2", {"ok": False})

    second = Progress("answers")  # simulates a new run
    assert "q1" in second and second.get("q2") == {"ok": False} and len(second) == 2
    assert "q3" not in second


def test_progress_is_discarded_when_the_code_that_affects_answers_has_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_DIR", tmp_path)
    monkeypatch.setattr(progress, "code_fingerprint", lambda: "version-1")
    Progress("answers").put("q1", {"ok": True})

    monkeypatch.setattr(progress, "code_fingerprint", lambda: "version-2")  # the prompt or graph changed
    assert "q1" not in Progress("answers")


def test_fresh_ignores_saved_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_DIR", tmp_path)
    Progress("cost").put("a", 1)
    assert len(Progress("cost", fresh=True)) == 0
    assert len(Progress("cost")) == 1  # the file is not deleted until new entries are written


def test_parts_are_stored_independently(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_DIR", tmp_path)
    Progress("answers").put("q1", 1)
    assert "q1" not in Progress("guardrails")
