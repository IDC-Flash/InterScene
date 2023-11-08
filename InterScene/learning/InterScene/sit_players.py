# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import torch 
import json
import datetime

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import learning.amp_players as amp_players

class SitPlayer(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)
        return

    def restore(self, fn):
        super().restore(fn)
        self.checkpoint_fn = fn
        return

    def run(self):
        is_determenistic = self.is_determenistic
        num_envs = self.env.num_envs
        num_trials = num_envs
        assert num_envs == num_trials
        num_repeat = 3

        print("evaluating sit policy: {} trials".format(num_envs))

        eval_res = {}

        for i in range(num_repeat):
            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)
            
            done_indices = []
            normal_done_indices = []

            games_played = 0
            games_success = 0
            sum_success_executionTime = 0
            sum_success_precision = 0

            games_fail_because_terminate = 0
            games_fail = 0

            has_collected = torch.zeros(num_envs, device=self.device)

            while games_played < num_trials:
                obs_dict = self.env_reset(normal_done_indices)
                action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info =  self.env_step(self.env, action)

                normal_all_done_indices = done.nonzero(as_tuple=False)
                normal_done_indices = normal_all_done_indices[::self.num_agents, 0]

                done = torch.where(has_collected == 0, done, torch.zeros_like(done))
                has_collected[done == 1] += 1

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents, 0]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    success_done = info['success'][done_indices]
                    executionTime_done = info['executionTime'][done_indices]
                    percision_done = info['precision'][done_indices]
                    terminate_done = info['terminate'][done_indices]

                    # compute number of success
                    success_indices = success_done.nonzero(as_tuple=False)[:, 0]
                    success_count = len(success_indices)
                    games_success += success_count

                    terminate_count = torch.logical_and(terminate_done, success_done == 0).sum().cpu().item()
                    fail_count = done_count - success_count - terminate_count
                    
                    games_fail_because_terminate += terminate_count
                    games_fail += fail_count

                    if success_count > 0:
                        sum_success_executionTime += executionTime_done[success_indices].sum().cpu().numpy()
                        sum_success_precision += percision_done[success_indices].sum().cpu().numpy()
  
                # self._post_step(info)
            
            success_rate = games_success / games_played
            if games_success > 0:
                mean_success_executionTime = sum_success_executionTime / games_success
                mean_success_precision = sum_success_precision / games_success
            else:
                mean_success_executionTime = 0
                mean_success_precision = 0
            
            curr_exp_name = "ObjectSet_{}_{}".format("test", i)
            eval_res[curr_exp_name] = {
                "num_trials": games_played,
                "success_trials": games_success,
                "success_rate": success_rate,
                "success_executionTime": mean_success_executionTime,
                "success_precision": mean_success_precision,
                "fail_trials": games_fail,
                "fail_trials_because_terminate": games_fail_because_terminate,
                "fail_trials_total": games_fail + games_fail_because_terminate,
            }

            print(curr_exp_name)
            print(eval_res[curr_exp_name])

        # save metrics
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = os.path.dirname(self.checkpoint_fn)
        ckp_name = os.path.basename(self.checkpoint_fn)
        save_dir = os.path.join(folder_name, ckp_name[:-4] + "_metrics")
        os.makedirs(save_dir, exist_ok=True)
        json.dump(eval_res, open(os.path.join(save_dir, "metrics_{}.json".format(time)), 'w'))
        print("save at {}".format(os.path.join(save_dir, "metrics_{}.json".format(time))))

        return
