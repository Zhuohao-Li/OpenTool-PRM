# OpenTool-PRM

## Roadmap

[ ] skywork-v2 as the rm

[ ] polar as the rm

[ ] GRPO supported

[x] Qwen3-235B-A22B as the juding supported

[x] Gemini-2.5-flash as the juding supported

[x] multi-thread judging with critic supported

[x] predefined prompts

## Get started

Ft needs a search engine setup or a local retriever to support searching.

For local retrieval, run `bash retrieval_launch.sh`

For rlab search engine, run `bash server_launch.sh`

After this, in another terminal, run:

```
# Note: make sure you are in the root path
# Note: make sure to change the n_gpus_per_node and CUDA_VISIBLE_DEVICES in train_ppo.sh
export PROJECT_ENV=local
bash train_ppo.sh
```

## about the repo

* `main`
* `zhuohao-feat`: stable version for running with llm as judge with search engine and local retrieval
* `zhuohao-sglang`: use SGLang as the rollout backend (ongoing)
* `zhuohao-skywork`: use Skywork-v2 as the reward model to score process of rollout

**key files**

* `train_ppo.sh`, `train_grpo.sh`
* `server_launch.sh`, `retrieval_launch.sh`
* `verl/utils/reward_score`: reward model, `pc_reward.py`: llm as judge reward, `skywork_reward.py`: use Skywork-v2 as rm


We thank [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [veRL](https://github.com/volcengine/verl), and [ReasonFlux](https://github.com/Gen-Verse/ReasonFlux)