"""Domain-dispatching rollout for mixed Math + Search + Tau on-policy distillation.

slime allows a single ``--custom-generate-function-path``. To roll out three domains in
one job, this dispatcher routes each prompt to its NATIVE rollout based on
``sample.metadata["domain"]``:

    "math"   -> slime.rollout.sglang_rollout.generate   (STOCK single-turn rollout)
    "search" -> generate_with_search_tools_qwen_sft_no_drift.generate
                (student turns -> local sglang router; search tool -> retriever @ 131.179.168.117:8000)
    "tau"    -> generate_with_tau.generate
                (student turns -> local sglang router; user turns -> OPENAI_API_BASE)

The search + tau example dirs must be on PYTHONPATH (the run script adds
examples/search-r1 and examples/tau-bench). ``math`` needs no extra dir — the stock
single-turn rollout lives in slime itself.

reward=None INVARIANT. The OPD teacher RM (opd_multi_teacher.reward_func) is only
invoked by generate_and_rm when ``sample.reward is None`` (async_rm prefers
--custom-rm-path, but only fires on an unset reward). So every branch here must leave
``sample.reward = None``:
  * math   : the stock rollout never sets a reward -> fine as-is.
  * search : generate_with_search_tools...generate leaves reward=None -> fine as-is.
  * tau    : generate_with_tau.generate ALWAYS sets sample.reward to the tau task
             reward (a float). Left set, it would (a) suppress the teacher RM and
             (b) feed a float where post_process expects the teacher's logprob JSON.
             So we CLEAR it (mirrors tau2if/generate_with_tau_opd.generate).
"""

import logging

# math: stock single-turn rollout (in slime; no example dir needed).
from slime.rollout.sglang_rollout import generate as _generate_math

# Requires examples/search-r1 and examples/tau-bench on PYTHONPATH.
from generate_with_search_tools_qwen_sft_no_drift import generate as _generate_search
from generate_with_tau import generate as _generate_tau

logger = logging.getLogger(__name__)


async def generate(args, sample, sampling_params):
    domain = (sample.metadata or {}).get("domain")
    if domain == "math":
        return await _generate_math(args, sample, sampling_params)
    if domain == "search":
        return await _generate_search(args, sample, sampling_params)
    if domain == "tau":
        sample = await _generate_tau(args, sample, sampling_params)
        # generate_with_tau builds a FRESH Sample(metadata=res.info) (res_to_sample),
        # so the mixed-dataset row's metadata — including the "domain" tag — is DROPPED.
        # The OPD teacher RM (opd_multi_teacher.reward_func) then reads
        # sample.metadata["domain"] to route to teacher_tau and, finding it None, raises
        # (this crashed job <JOB_ID> at first-rollout reward time). Math/search mutate the
        # input sample in place so they keep the tag; only tau needs re-stamping. Merge so
        # tau's env info (res.info) is preserved alongside the restored domain.
        sample.metadata = {**(sample.metadata or {}), "domain": "tau"}
        # pure OPD: drop the tau task reward so the OPD teacher RM scores the tokens.
        sample.reward = None
        return sample
    raise ValueError(
        f"sample.metadata['domain'] must be one of ['math', 'search', 'tau'], got {domain!r}. "
        "Tag every row in the mixed dataset (see prepare_opd_mixed_data.py)."
    )
