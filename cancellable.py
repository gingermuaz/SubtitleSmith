"""Running slow work so that Cancel doesn't have to wait for it."""
import threading

from errors import ProcessCancelled


def run_cancellable(work, cancel_check):
    """Runs work() on a helper thread so Cancel doesn't have to wait for it to finish.

    On Cancel this raises ProcessCancelled straight away. work() carries on in the background and its
    result is dropped (except a model that finishes loading, which is kept for next time).
    """
    outcome = {}

    def run():
        try:
            outcome["result"] = work()
        except BaseException as e:  # handed back to the caller below
            outcome["error"] = e

    helper = threading.Thread(target=run, daemon=True)
    helper.start()
    while helper.is_alive():
        helper.join(0.2)
        if cancel_check and cancel_check() and helper.is_alive():
            raise ProcessCancelled()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]
