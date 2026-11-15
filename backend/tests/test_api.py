"""End-to-end tests through the HTTP API, including the decision graph's three abstain branches."""

from .conftest import fetch, signal_of


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert len(meta["grasps"]) == 6 and len(meta["sources"]) == 5
    assert "calibration_curve" in client.get("/api/evaluation").json()


def test_uncalibrated_wearer_is_never_acted_on(client):
    client.post("/api/users", json={"user_id": "fresh", "source": "wearer_d"})
    for _ in range(5):
        rec = fetch(client, "fresh")
        out = client.post("/api/predict", json={"user_id": "fresh", **signal_of(rec)}).json()
        assert out["decision"] == "abstain" and out["abstain_reason"] == "not_calibrated"
        assert out["predicted_grasp"] is not None        # the guess is shown, just not acted on


def test_calibrated_wearer_acts_and_is_accurate(client):
    client.post("/api/users", json={"user_id": "fitted", "source": "wearer_e"})
    profile = client.post("/api/calibrate", json={"user_id": "fitted", "reps": 2}).json()
    assert profile["calibrated_grasps"] == 6 and profile["recordings_confirmed"] == 12
    acted = correct = 0
    for _ in range(40):
        rec = fetch(client, "fitted")
        out = client.post("/api/predict", json={"user_id": "fitted", **signal_of(rec)}).json()
        if out["decision"] == "act":
            acted += 1
            correct += out["predicted_grasp"] == rec["true_grasp"]
            assert out["confidence"] >= out["confidence_floor"]
            assert len(out["contributions"]) == 12 and out["reasoning"]
        else:
            assert out["abstain_reason"] == "low_confidence"
    assert acted >= 20 and correct / acted > 0.85


def test_faulty_recording_blocks_prediction_and_learning(client):
    client.post("/api/users", json={"user_id": "faulty", "source": "wearer_a"})
    client.post("/api/calibrate", json={"user_id": "faulty", "reps": 1})
    before = client.get("/api/profile/faulty").json()["recordings_confirmed"]
    for fault in ["dropout", "spike", "hum"]:
        rec = fetch(client, "faulty", inject_fault=fault)
        out = client.post("/api/predict", json={"user_id": "faulty", **signal_of(rec)}).json()
        assert out["decision"] == "abstain" and out["abstain_reason"] == "sensor_fault"
        assert out["predicted_grasp"] is None and fault in [a["type"] for a in out["anomalies"]]
        fb = client.post("/api/feedback", json={"user_id": "faulty", "actual_grasp": rec["true_grasp"],
                                                **signal_of(rec)}).json()
        assert fb["learned"] is False
    assert client.get("/api/profile/faulty").json()["recordings_confirmed"] == before


def test_feedback_logs_outcome_then_learns(client):
    client.post("/api/users", json={"user_id": "learner", "source": "wearer_b"})
    rec = fetch(client, "learner", grasp="hook")
    fb = client.post("/api/feedback", json={"user_id": "learner", "actual_grasp": "hook", **signal_of(rec)}).json()
    assert fb["learned"] is True and fb["recordings_confirmed"] == 1
    assert fb["recordings_learned"]["hook"] == 1 and fb["grasp_distribution"]["hook"] == 1


def test_queue_never_repeats_a_recording_within_a_cycle(client):
    client.post("/api/users", json={"user_id": "queue", "source": "wearer_c"})
    seen = set()
    for _ in range(60):
        rec = fetch(client, "queue")
        key = (rec["true_grasp"], rec["recording_index"])
        assert key not in seen
        seen.add(key)


def test_profiles_are_isolated_and_resettable(client):
    client.post("/api/users", json={"user_id": "one", "source": "wearer_a"})
    client.post("/api/users", json={"user_id": "two", "source": "wearer_a"})
    client.post("/api/calibrate", json={"user_id": "one", "reps": 1})
    assert client.get("/api/profile/two").json()["recordings_confirmed"] == 0
    assert client.post("/api/profile/one/reset").json()["recordings_confirmed"] == 0
    ids = [u["user_id"] for u in client.get("/api/users").json()["users"]]
    assert "one" in ids and "two" in ids


def test_bad_input_is_rejected(client):
    assert client.post("/api/predict", json={"user_id": "x", "ch1": [0.1] * 10, "ch2": [0.1] * 10}).status_code == 400
    assert client.post("/api/predict", json={"user_id": "x", "ch1": [0.1] * 300, "ch2": [0.1] * 299}).status_code == 400
    assert client.post("/api/signal/next", json={"user_id": "x", "grasp": "nope"}).status_code == 400
    assert client.post("/api/signal/next", json={"user_id": "x", "inject_fault": "nope"}).status_code == 400
    assert client.post("/api/users", json={"user_id": "x", "source": "nobody"}).status_code == 400
    assert client.post("/api/feedback", json={"user_id": "x", "actual_grasp": "nope", "ch1": [0.0] * 300,
                                              "ch2": [0.0] * 300}).status_code == 400
    assert client.post("/api/predict", json={"user_id": "x", "ch1": [0.1] * 20001, "ch2": [0.1] * 20001}).status_code == 422


def test_report_carries_expected_accuracy(client):
    report = client.get("/api/report/fitted").json()
    assert report["expected_accuracy_by_reps"]["0"] < 40 < report["expected_accuracy_by_reps"]["3"]
