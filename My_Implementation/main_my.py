import copy

from agent_my import CNetAgentMy
from arena_my import ArenaMy
from switch_cnet_my import SwitchCNetMy
from switch_game_my import SwitchGameMy


opts = {

   "game":"switch",
   "game_nagents":3,
   "game_action_space":2,
   "game_comm_limited":True,
   "game_comm_bits":1,
   "game_comm_sigma":2,
   "nsteps":6,
   "gamma":1,
   "model_dial":False,
   "model_target":True,
   "model_bn":True,
   "model_know_share":True,
   "model_action_aware":True,
   "model_rnn_size":128,
   "model_rnn_layers":2,
   "model_rnn_dropout_rate":0,
   "bs":2,
   "learningrate":0.00005,
   "momentum":0.05,
   "eps":0.05,
   "eps_decay":1.0,
   "nepisodes":10001,
   "step_test":10,
   "step_target":100,
   "cuda":0
}

opts["game_action_space_total"] = opts["game_action_space"] + opts["game_comm_bits"]
opts["game_comm_bits"] = 2**opts["game_comm_bits"]
game = SwitchGameMy(opts)
cnet = SwitchCNetMy(opts)
cnet_target = copy.deepcopy(cnet)
agents = [None]

for i in range(1, opts["game_nagents"]+1):
    agents.append(CNetAgentMy(opts, game=game, model=cnet, target=cnet_target, index=i))

arena = ArenaMy(opts, game)

arena.train(agents)
