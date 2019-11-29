import numpy as np
import torch

class SwitchGameMy:

    def __init__(self, opt):

        self.game_actions = {
            "Nothing": 0,
            "Tell": 1
        }
        self.game_status = {
            "Outside": 0,
            "Inside": 1
        }
        self.opt = opt

        self.reward_all_live = 1
        self.reward_all_die = -1

        self.reset()

    def reset(self):

        self.step_count = 0
        self.reward = torch.zeros(self.opt["game_nagents"])
        self.has_been = torch.zeros(self.opt["nsteps"], self.opt["game_nagents"])
        self.terminal = 0
        self.active_agent = torch.zeros(self.opt['nsteps'], dtype=torch.long)  # 1-indexed agents

        for step in range(self.opt['nsteps']):

            agent_id = 1 + np.random.randint(self.opt['game_nagents'])
            self.active_agent[step] = agent_id
            self.has_been[step][agent_id - 1] = 1

        return self

    def get_action_range(self, step, agent_id):

        comm_range = torch.zeros(2)

        if self.active_agent[step] == agent_id:
            action_range = torch.tensor([1, self.opt["game_action_space"]])
            comm_range = torch.tensor([self.opt["game_action_space"], self.opt["game_action_space_total"]])

        else:
            action_range = torch.tensor([1, 1])

        return action_range, comm_range

    def get_comm_limited(self, step, agent_id):
        if self.opt['game_comm_limited']:
            comm_lim = 0
            if step > 0 and agent_id == self.active_agent[step]:
                comm_lim = self.active_agent[step-1]
            return comm_lim
        return None

    def get_reward(self, a_t):
        active_agent_idx = self.active_agent[self.step_count] - 1
        if a_t[active_agent_idx] == self.game_actions["Tell"] and not self.terminal:
            has_been = self.has_been[:self.step_count+1].sum()
            if has_been == self.opt["game_nagents"]:
                self.reward = self.reward_all_live;
            else:
                self.reward = self.reward_all_die;
            self.terminal = 1
        elif self.step_count == self.opt["nsteps"] - 1:
            self.terminal = 1

        return self.reward, self.terminal

    def step(self, a_t):
        reward, terminal = self.get_reward(a_t)
        self.step_count += 1
        return reward, terminal

    def get_state(self):

        state = torch.zeros(self.opt['game_nagents'], dtype=torch.long)

        for a in range(1, self.opt['game_nagents']):
            if self.active_agent[self.step_count] == a:
                state[a-1] = self.game_status["Inside"]

        return state

    # def god_strategy_reward(self, steps):
    #     reward = 0
    #     has_been = self.has_been[:self.opt['nsteps']].sum()
    #     if has_been == self.opt['nagents']:
    #         reward = self.reward_all_die
    #
    #     return reward
