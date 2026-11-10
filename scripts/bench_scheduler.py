"""Count cache preemptions without changing the Ascend scheduling policy."""

from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler


class BenchmarkScheduler(BalanceScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("HIF8_BENCH_PREEMPTION_TRACKING=1", flush=True)

    def _preempt_request(self, request, timestamp):
        print("HIF8_BENCH_PREEMPTION=1", flush=True)
        return super()._preempt_request(request, timestamp)
