from __future__ import annotations

from backend.app.options.pipeline_reliability import classify_fetch_failure, ended_on_failure_streak


class TestClassifyFetchFailure:
    def test_recognizes_the_exact_2026_09_28_signature(self):
        detail = ("expiry_list did not return status=success: {'status': 'failure', 'remarks': "
                  "{'error_code': None, 'error_type': None, 'error_message': None}, 'data': ''}")
        assert classify_fetch_failure(detail) == "AUTH_FAILURE_SUSPECTED"

    def test_recognizes_the_signature_regardless_of_key_order(self):
        detail = "remarks: {'error_type': None, 'error_code': None, 'error_message': None}"
        assert classify_fetch_failure(detail) == "AUTH_FAILURE_SUSPECTED"

    def test_does_not_misclassify_a_real_error_code_as_auth_suspected(self):
        detail = "option_chain did not return status=success: {'status': 'failure', 'remarks': {'error_code': 'DH-905', 'error_type': 'Invalid_Request', 'error_message': 'bad segment'}}"
        assert classify_fetch_failure(detail) != "AUTH_FAILURE_SUSPECTED"

    def test_timeout_signature(self):
        assert classify_fetch_failure("HTTPSConnectionPool: Read timed out.") == "TIMEOUT_SUSPECTED"
        assert classify_fetch_failure("requests.exceptions.ConnectTimeout") == "TIMEOUT_SUSPECTED"

    def test_connection_failure_signature(self):
        assert classify_fetch_failure("ConnectionError: Max retries exceeded with url") == "CONNECTION_FAILURE_SUSPECTED"
        assert classify_fetch_failure("Failed to establish a new connection: [Errno 11001]") == "CONNECTION_FAILURE_SUSPECTED"

    def test_unknown_fallback_never_guesses_beyond_the_text(self):
        assert classify_fetch_failure("option_chain did not return status=success: {'status': 'failure', 'remarks': 'Internal Server Error'}") == "PROVIDER_FAILURE_UNKNOWN"

    def test_empty_string_is_unknown_not_a_crash(self):
        assert classify_fetch_failure("") == "PROVIDER_FAILURE_UNKNOWN"


class TestEndedOnFailureStreak:
    def test_false_when_threshold_is_none(self):
        assert ended_on_failure_streak(["FETCH_FAILURE"] * 10, threshold=None) is False

    def test_false_when_fewer_outcomes_than_threshold(self):
        assert ended_on_failure_streak(["FETCH_FAILURE", "FETCH_FAILURE"], threshold=5) is False

    def test_true_when_last_n_are_all_fetch_failure(self):
        outcomes = ["PERSISTED", "PERSISTED", "FETCH_FAILURE", "FETCH_FAILURE", "FETCH_FAILURE"]
        assert ended_on_failure_streak(outcomes, threshold=3) is True

    def test_false_when_a_success_interrupts_the_recent_tail(self):
        outcomes = ["FETCH_FAILURE", "FETCH_FAILURE", "PERSISTED", "FETCH_FAILURE", "FETCH_FAILURE"]
        assert ended_on_failure_streak(outcomes, threshold=3) is False

    def test_true_when_session_ended_entirely_on_failures(self):
        assert ended_on_failure_streak(["FETCH_FAILURE"] * 6, threshold=6) is True
