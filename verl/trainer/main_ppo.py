# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():  # TODO: add llm_judge related statistic in val log
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., include_llm_judge_score=False, llm_judge_config=None, include_skywork_score=False, skywork_config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.include_llm_judge_score = include_llm_judge_score
        self.include_skywork_score = include_skywork_score
        self.rollout_from_generation_loop = False

        if include_llm_judge_score and llm_judge_config is not None:
            self.llm_judge_model_id = llm_judge_config.model_id
            self.rollout_from_generation_loop = llm_judge_config.rollout_from_generation_loop
            # Parallel processing configuration
            self.max_workers = llm_judge_config.max_workers
            self._lock = threading.Lock()  # Thread lock for shared resources

        if include_skywork_score and skywork_config is not None:
            self.rollout_from_generation_loop = skywork_config.rollout_from_generation_loop
            self.max_workers = llm_judge_config.max_workers
            self._lock = threading.Lock()  # Thread lock for shared resources

    def _process_single_llm_judge(self, rollout_result, reward_tensor):
        """Process LLM judge for a single rollout.
        
        Args:
            rollout_result: single rollout result
            reward_tensor: in-place operation
            
        Returns:
            dict: LLM judge result
        """
        i = rollout_result['index']
        prompt_str = rollout_result['prompt_str']
        response_str = rollout_result['response_str']

        from verl.utils.reward_score.pc_reward import compute_score_llm_judge, filter_llm_judge_score, write_tag_rewards

        try:
            score_llm_result = compute_score_llm_judge(
                query_str=prompt_str, 
                solution_str=response_str, 
                model_id=self.llm_judge_model_id, 
                max_score=1.
            )
            
            # Process LLM judge score using the new organized function
            is_score_valid = filter_llm_judge_score(score_llm_result, response_str, self.tokenizer)
            
            if is_score_valid:
                # Apply LLM judge rewards directly here
                write_tag_rewards(
                    batch_idx=rollout_result['index'],
                    reward_tensor=reward_tensor,
                    response_str=response_str,
                    search_scores=score_llm_result.get('search_score', []),
                    think_scores=score_llm_result.get('think_score', []),
                    tokenizer=self.tokenizer,
                    max_score_extracted=score_llm_result.get('max_score_extracted', 1),
                )
                return 1, 1
            else:
                return 1, 0
        except Exception as e:
            print(f"[WARN] LLM judge API call failed for rollout {i}: {e}")
            return 1, 0

    def _process_single_llm_judge_per_turn(self, rollout_result, reward_tensor):
        """Process LLM judge for a single rollout.
        
        Args:
            rollout_result: single rollout result
            reward_tensor: in-place operation
            
        Returns:
            dict: LLM judge result
        """
        i = rollout_result['index']
        prompt_str = rollout_result['prompt_str']
        response_str = rollout_result['response_str']
        
        from verl.utils.reward_score.pc_reward_turn import compute_score_llm_judge, filter_llm_judge_score, write_tag_rewards

        try:
            score_llm_result = compute_score_llm_judge(
                query_str=prompt_str, 
                solution_str=response_str, 
                model_id=self.llm_judge_model_id
            )
            
            # Process LLM judge score using the new organized function
            is_score_valid = filter_llm_judge_score(score_llm_result)
            
            if is_score_valid:
                write_tag_rewards(
                    reward_tensor=reward_tensor,
                    batch_idx=rollout_result['index'],
                    turn_pos=rollout_result['turn_pos'],
                    score=score_llm_result['score'],
                    max_score_extracted=score_llm_result['max_score']
                )
                return 1, 1
            else:
                return 1, 0
        except Exception as e:
            print(f"[WARN] LLM judge API call failed for rollout {i}: {e}")
            return 1, 0

    def _process_single_skywork_per_turn_batch(self, rollout_results, reward_tensor):
        """Batch process Skywork reward model for all rollout_results (per turn).
        Args:
            rollout_results: list of dicts, each with 'index', 'turn_pos', 'prompt_str', 'response_str'
            reward_tensor: in-place operation
        Returns:
            (total_count, valid_count)
        """
        from verl.utils.reward_score.skywork_reward import compute_skywork_reward_in_batch, write_tag_rewards
        total_count = len(rollout_results)
        valid_count = 0
        if total_count == 0:
            return 0, 0

        conversations = [
            [
                {"role": "user", "content": rollout_result['prompt_str']},
                {"role": "assistant", "content": rollout_result['response_str']}
            ]
            for rollout_result in rollout_results
        ]

        try:
            scores = compute_skywork_reward_in_batch(conversations)
        except Exception as e:
            print(f"[WARN] Skywork batch reward computation failed: {e}")
            return total_count, 0

        for rollout_result, score in zip(rollout_results, scores):
            try:
                write_tag_rewards(
                    reward_tensor=reward_tensor,
                    batch_idx=rollout_result['index'],
                    turn_pos=rollout_result['turn_pos'],
                    score=score
                )
                valid_count += 1
            except Exception as e:
                print(f"[WARN] Failed to write Skywork reward for index {rollout_result['index']} turn {rollout_result['turn_pos']}: {e}")
        return total_count, valid_count

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        
        # Simple LLM judge tracking
        total_judge_count = 0
        valid_judge_count = 0
        
        # Skywork tracking
        total_skywork_count = 0
        valid_skywork_count = 0

        # Process all rollouts first
        rollout_results = []
        for i in range(len(data)):
            data_item = data[i]
            
            # Process single rollout (reward computation, etc.)
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)

            if self.include_llm_judge_score and self.rollout_from_generation_loop:   # generate multi-turn qa pair
                llm_judge_func = self._process_single_llm_judge_per_turn
                response_positions = data_item.meta_info['response_id_positions_per_turn'][i]
                last_turn_pos = 0

                ori_q = re.search(r'Question:\s*(.*?)(?:\n|<\|im_end\|>|$)', prompt_str, re.DOTALL)
                ori_q = ori_q.group(1).strip()

                prompt_str_per_turn = ori_q
                response_str = ''
                for turn_idx in range(len(response_positions)):
                    response_str_per_turn = self.tokenizer.decode(valid_response_ids[last_turn_pos:response_positions[turn_idx]+1])
                    last_turn_pos = response_positions[turn_idx]+1
                    rollout_result = {
                        'index': i,
                        'turn_index': turn_idx,
                        'turn_pos': response_positions[turn_idx],
                        'prompt_str': prompt_str_per_turn,
                        'response_str': response_str_per_turn
                    }
                    if turn_idx<len(response_positions)-1:  # dont judge last <answer> turn, leave it to orm em
                        rollout_results.append(rollout_result)
                    prompt_str_per_turn = prompt_str_per_turn + response_str_per_turn
                    response_str += response_str_per_turn

            elif self.include_skywork_score and self.rollout_from_generation_loop:
                response_positions = data_item.meta_info['response_id_positions_per_turn'][i]
                response_str = self.tokenizer.decode(valid_response_ids)
                last_turn_pos = 0

                ori_q = re.search(r'Question:\s*(.*?)(?:\n|<\|im_end\|>|$)', prompt_str, re.DOTALL)
                ori_q = ori_q.group(1).strip()

                prompt_str_per_turn = ori_q
                response_str = ''
                for turn_idx in range(len(response_positions)):
                    response_str_per_turn = self.tokenizer.decode(valid_response_ids[last_turn_pos:response_positions[turn_idx]+1])
                    last_turn_pos = response_positions[turn_idx]+1
                    rollout_result = {
                        'index': i,
                        'turn_index': turn_idx,
                        'turn_pos': response_positions[turn_idx],
                        'prompt_str': prompt_str_per_turn,
                        'response_str': response_str_per_turn
                    }
                    if turn_idx<len(response_positions)-1:  # dont judge last <answer> turn, leave it to orm em
                        rollout_results.append(rollout_result)
                    prompt_str_per_turn = prompt_str_per_turn + response_str_per_turn
                    response_str += response_str_per_turn
            else:
                llm_judge_func = self._process_single_llm_judge
                response_str = self.tokenizer.decode(valid_response_ids)
                rollout_result = {
                    'index': i,
                    'prompt_str': prompt_str,
                    'response_str': response_str
                }
                rollout_results.append(rollout_result)
            

            sequences_str = prompt_str + response_str
            
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
            # Update reward tensor
            reward_tensor[i, valid_response_length - 1] = score

            # Handle printing logic
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)


        # Process LLM judge in parallel if enabled
        if self.include_llm_judge_score:
            max_workers=self.max_workers
            print(f"Processing {len(rollout_results)} LLM judge calls in parallel with {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_rollout = {
                    executor.submit(llm_judge_func, rollout, reward_tensor): rollout 
                    for rollout in rollout_results
                }
                
                for future in as_completed(future_to_rollout):
                    try:
                        total_count, valid_count = future.result()
                        total_skywork_count += total_count
                        valid_skywork_count += valid_count
                    except Exception as e:
                        print(f"[WARN] LLM judge failed: {e}")
                        # Continue processing other results


        if self.include_skywork_score:
            max_workers=self.max_workers
            print(f"Trigger Skywork reword model in batch size {len(rollout_results)}")
            total_count, valid_count = self._process_single_skywork_per_turn_batch(rollout_results, reward_tensor)
            total_skywork_count += total_count
            valid_skywork_count += valid_count

           
        # Save metrics
        if self.include_llm_judge_score and total_judge_count > 0:
            valid_rate = valid_judge_count / total_judge_count
            print(f"[LLM Judge Valid Rate] {valid_judge_count}/{total_judge_count} = {valid_rate:.3f}")
            
            # Save to data for metrics logging
            if not hasattr(data, 'llm_judge_metrics'):
                data.llm_judge_metrics = {}
            data.llm_judge_metrics['llm_judge_valid_rate'] = valid_rate
            data.llm_judge_metrics['llm_judge_valid_count'] = valid_judge_count
            data.llm_judge_metrics['llm_judge_total_count'] = total_judge_count

        # Save Skywork metrics
        if self.include_skywork_score and total_skywork_count > 0:
            valid_rate = valid_skywork_count / total_skywork_count
            print(f"[Skywork Reward Valid Rate] {valid_skywork_count}/{total_skywork_count} = {valid_rate:.3f}")
            
            # Save to data for metrics logging
            if not hasattr(data, 'skywork_metrics'):
                data.skywork_metrics = {}
            data.skywork_metrics['skywork_valid_rate'] = valid_rate
            data.skywork_metrics['skywork_valid_count'] = valid_skywork_count
            data.skywork_metrics['skywork_total_count'] = total_skywork_count

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, include_llm_judge_score=config.llm_as_judge.include_score, 
                            llm_judge_config=config.llm_as_judge, include_skywork_score=config.skywork_reward.include_score, 
                            skywork_config=config.skywork_reward)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, include_llm_judge_score=config.llm_as_judge.include_score_val, 
                            llm_judge_config=config.llm_as_judge, include_skywork_score=config.skywork_reward.include_score_val, 
                            skywork_config=config.skywork_reward)
        
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
