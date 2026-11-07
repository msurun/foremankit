"""Check that verification handoffs retain their original retry policy.

Taking over failed work does not establish reviewer independence.
"""
from unittest.mock import Mock, patch

from foremankit.postcheck import verify_with_handoffs
from foremankit.records import RouterError, TaskRequest


def test_handoffs_exclude_only_the_failed_source():
    """Retrying work may use another provider from the same company."""
    router = Mock(routing={"verify": {"handoffs": 1}})
    router.run_chain.side_effect = RouterError("no source")
    candidate = Mock(key="agy")
    result = {"_cand": candidate, "verify": {}}
    verification = Mock(report={"passed": False}, failure="check failed")
    with patch("foremankit.postcheck.VerifyRun", return_value=verification):
        verify_with_handoffs(router, result, TaskRequest("write code", "quick"), Mock())
    assert router.run_chain.call_args.kwargs["exclude"] == ("agy",)
