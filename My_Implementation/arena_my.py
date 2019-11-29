import copy

import torch
from torch.autograd import Variable


class ArenaMy:

    def __init__(self, opt, game):
        self.opt = opt
        self.game = game
        self.eps = opt["eps"]

    def create_episode(self):
        opt = self.opt
        episode = {}
        episode["steps"] = 0
        episode["ended"] = 0
        episode["r"] = torch.zeros(opt["game_nagents"])
        episode["step_records"] = []

        return episode

    def create_step_record(self):

        opt = self.opt
        record = {}
        record["s_t"] = None
        record["r_t"] = torch.zeros(opt["game_nagents"])
        record["terminal"] = 0
        record["agent_inputs"] = []
        record["a_t"] = torch.zeros(opt["game_nagents"])
        record["a_comm_t"] = torch.zeros(opt["game_nagents"])
        record["comm"] = torch.zeros(opt["game_nagents"], opt["game_comm_bits"])
        record["comm_target"] = torch.zeros(opt["game_nagents"], opt["game_comm_bits"])
        record["hidden"] = torch.zeros(opt["game_nagents"], opt["model_rnn_layers"], opt["model_rnn_size"])
        record["hidden_target"] = torch.zeros(opt["game_nagents"], opt["model_rnn_layers"], opt["model_rnn_size"])
        record["q_a_t"] = torch.zeros(opt["game_nagents"])
        record["q_a_max_t"] = torch.zeros(opt["game_nagents"])
        record["q_comm_t"] = torch.zeros(opt["game_nagents"])
        record["q_comm_max_t"] = torch.zeros(opt["game_nagents"])

        return record

    def run_episode(self, agents, train_mode=False):

        opt = self.opt
        game = self.game
        game.reset()
        self.eps = self.eps * opt["eps_decay"]
        step = 0
        episode = self.create_episode()
        s_t = game.get_state()
        episode["step_records"].append(self.create_step_record())
        episode["step_records"][0]["s_t"] = s_t
        episode_steps = train_mode and opt["nsteps"]

        while step < episode_steps:

            episode["step_records"].append(self.create_step_record())

            for i in range(1, opt['game_nagents'] + 1):

                agent = agents[i]
                agent_idx = i - 1

                comm_limited = game.get_comm_limited(step, agent.id)
                comm = episode["step_records"][step]["comm"].clone()[comm_limited - 1]

                # Get Prev action

                prev_action = 1
                prev_message = 1
                if step > 0 and episode["step_records"][step - 1]["a_t"][agent_idx] > 0:
                    prev_action = episode["step_records"][step - 1]["a_t"][agent_idx]
                if step > 0 and episode["step_records"][step - 1]["a_comm_t"][agent_idx] > 0:
                    prev_message = episode["step_records"][step - 1]["a_comm_t"][agent_idx]
                prev_action = (prev_action, prev_message)

                batch_agent_index = agent_idx

                # print("Hidden size")
                # print(episode["step_records"][step]["hidden"][0].shape)
                # print(episode["step_records"][step])
                agent_inputs = {
                    's_t': episode["step_records"][step]["s_t"][agent_idx],
                    'messages': comm,
                    'hidden': episode["step_records"][step]["hidden"][agent_idx, :, :],  # Hidden state
                    'prev_action': prev_action,
                    'agent_index': batch_agent_index
                }

                episode["step_records"][step]["agent_inputs"].append(agent_inputs)
                #
                # print("Model")
                # print(agent.model)
                #
                # print("Hidden")
                # print(episode["step_records"][step]["hidden"][agent_idx, :].shape)

                hidden_t, q_t = agent.model(**agent_inputs)
                episode["step_records"][step + 1]["hidden"][agent_idx] = hidden_t.squeeze()

                (action, action_value), (comm_vector, comm_action, comm_value) = agent.select_action_and_comm(step, q_t,
                                                                                                              eps=self.eps,
                                                                                                              train_mode=train_mode)
                episode["step_records"][step]["a_t"][agent_idx] = action
                episode["step_records"][step]["q_a_t"][agent_idx] = action_value
                # print("Step record episode")
                # print(episode["step_records"][step+1]["comm"])
                # print(comm_vector)
                # print(agent_idx)
                episode["step_records"][step + 1]["comm"][agent_idx] = comm_vector

                episode["step_records"][step]["a_comm_t"][agent_idx] = comm_action
                episode["step_records"][step]["q_comm_t"][agent_idx] = comm_value

            a_t = episode["step_records"][step]["a_t"]
            episode["step_records"][step]["r_t"], episode["step_records"][step]["terminal"] = self.game.step(a_t)

            if step < opt["nsteps"]:
                if not episode["ended"]:
                    episode["steps"] += 1
                    episode["r"] = episode["r"] + episode["step_records"][step]["r_t"]

                if episode["step_records"][step]["terminal"]:
                    episode["ended"] = 1

            if opt["model_target"] and train_mode:

                for i in range(1, opt["game_nagents"] + 1):
                    agent_target = agents[i]
                    agent_idx = i - 1

                    agent_inputs = episode["step_records"][step]["agent_inputs"][agent_idx]
                    comm_target = agent_inputs.get('messages', None)

                    agent_target_inputs = copy.copy(agent_inputs)
                    agent_target_inputs["messages"] = Variable(comm_target)
                    agent_target_inputs["hidden"] = episode["step_records"][step]["hidden_target"][agent_idx, :]
                    hidden_target_t, q_target_t = agent_target.model_target(**agent_target_inputs)
                    episode["step_records"][step + 1]["hidden_target"][agent_idx] = hidden_target_t.squeeze()

                    # Choose next arg max action and comm
                    (action, action_value), (comm_vector, comm_action, comm_value) = agent_target.select_action_and_comm(step, q_target_t, eps=0, target=True, train_mode=True)

                    # save target actions, comm, and q_a_t, q_a_max_t
                    episode["step_records"][step]["q_a_max_t"][agent_idx] = action_value
                    episode["step_records"][step + 1]["comm_target"][agent_idx] = comm_vector

            # Update step
            step = step + 1
            if episode["ended"] != 1:
                episode["step_records"][step]["s_t"] = self.game.get_state()
            else:
                break

        return episode

    def train(self, agents):

        opt = self.opt
        for agent in agents[1:]:
            agent.reset()
        reward_test = []
        num_episodes = opt["nepisodes"]
        reward_train = []
        for e in range(num_episodes):
            episode = self.run_episode(agents, train_mode=True)
            reward_train.append(episode["r"].sum()/opt["game_nagents"])
            if e % 32==0:
                print('train epoch:', e, 'avg steps:', episode["steps"], 'avg reward:', sum(reward_train)/32)
                reward_train = []
            agents[1].learn_from_episode(episode)

            # if e % opt["step_test"] == 0:
            #     episode = self.run_episode(agents, train_mode=False)
            #     reward_test.append(episode["r"])
            #     print('TEST EPOCH:', e, 'avg steps:', episode["steps"], 'avg reward:', episode["r"])
