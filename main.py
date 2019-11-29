import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_DAYS = 30

class Agent(nn.Module):

    # inputDim = 3, hiddenDim = 128, outputDim = 2
    def __init__(self, hiddenDim):
        super(Agent, self).__init__()
        self.hiddenDim = hiddenDim
        self.day_lookup = nn.Embedding(MAX_DAYS+1,hiddenDim)
        self.message_lookup = nn.Embedding(2,hiddenDim)
        self.agent_embedding = nn.Embedding(3,hiddenDim)
        # self.lstm=nn.GRU(hiddenDim, hiddenDim, num_layers=2, batch_first=True)
        self.outputLayer=nn.Linear(hiddenDim, 2)
        self.sigmoid=nn.Sigmoid()

    def forward(self, day, message, agent_id):
        day_embedding = self.day_lookup(day)
        message_embedding = self.message_lookup((message+0.5).long())
        agent_embedding = self.agent_embedding(agent_id)
        x = day_embedding + message_embedding + agent_embedding
        # input_lstm = input_lstm.view(1,1,self.hiddenDim)
        # lstmOut, _ = self.lstm(input_lstm)
        # T,B,D  = lstmOut.size(0),lstmOut.size(1) , lstmOut.size(2)
        # lstmOut = lstmOut.contiguous() 
        # lstmOut = lstmOut.view(B*T, D)
        action, message = self.sigmoid(self.outputLayer(x).view(-1))
        return action, message

def train(num_episodes, batch_size, agents):
    num_agents = len(agents)
    average_rewards = []
    for k in range(num_episodes):
        rewards = []
        for j in range(batch_size):
            episode_terminated = False
            message = torch.zeros(1, dtype=torch.long)
            action = torch.zeros(1, dtype=torch.long)
            day = 0
            agents_chosen = np.array([False for agent in agents])
            while not episode_terminated:
                day += 1
                agent_index = np.random.randint(num_agents)
                agents_chosen[agent_index] = True
                agent = agents[agent_index]
                day_tensor = torch.zeros(1, dtype=torch.long)
                day_tensor[0] = day
                agent_tensor = torch.zeros(1, dtype=torch.long)
                agent_tensor[0] = agent_index
                action, message = agent(day_tensor, message, agent_tensor)
                action, message = action.view(1, -1), message.view(1, -1)
                correct_action = 0
                reward = 0
                if action > 0.5 or day >= MAX_DAYS:
                    reward = -1
                    if np.all(agents_chosen):
                        correct_action = 1
                        reward = 1
                    rewards.append(reward)
                    episode_terminated = True
                criterion = nn.MSELoss()
                loss = criterion(action, torch.tensor([correct_action]).float())
                loss.backward()

        rewards = np.array(rewards)
        average_reward = rewards.mean()
        average_rewards.append(average_reward)
        print("Episode:", k+1, "Average Reward:", average_reward)
    return average_rewards

def print_episode(agents):
    num_agents = len(agents)
    episode_terminated = False
    message = torch.zeros(1, dtype=torch.long)
    action = torch.zeros(1, dtype=torch.long)
    day = 0
    while not episode_terminated:
        day += 1
        agent_index = np.random.randint(num_agents)
        print("Day:", day)
        print("Agent selected:", agent_index)
        agent = agents[agent_index]
        day_tensor = torch.zeros(1, dtype=torch.long)
        day_tensor[0] = day
        agent_tensor = torch.zeros(1, dtype=torch.long)
        agent_tensor[0] = agent_index
        action, message = agent(day_tensor, message, agent_tensor)
        # action, message = action.view(1, -1), message.view(1, -1)
        if message > 0.5:
            print("Agent toggled the bulb")
        else:
            print("Agent didn't toggle the bulb")
        if action > 0.5 or day >= MAX_DAYS:
            print("Agent Chose to TELL")
            episode_terminated = True
        else:
            print("Agent did not tell")
        print()


if __name__ == '__main__':
    num_agents = 3
    num_episodes = 100
    batch_size = 32
    agents = [Agent(128) for i in range(num_agents)]
    rewards = train(num_episodes, batch_size, agents)

    for i in range(10):
        print_episode(agents)
        print("\n\n\n")
