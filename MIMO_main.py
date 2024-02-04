from SLDAC import SLDAC_main
import matplotlib.pyplot as plt
import numpy as np
from SCAOPO import SCAOPO_main
from scipy.io import loadmat
from scipy.io import savemat
import argparse
from scipy.signal import hilbert

def get_envelop(x,n=None):
    analytic_signal = hilbert(x, N=n)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope
def main(args):
    SLDAC_reward_save1 = []
    SLDAC_cost_save1 = []
    SLDAC_reward_save2 = []
    SLDAC_cost_save2 = []
    SLDAC_reward_save3 = []
    SLDAC_cost_save3 = []
    SLDAC_reward_save4 = []
    SLDAC_cost_save4 = []
    SCAOPO_reward_save = []
    SCAOPO_cost_save = []
    for seed in range(1, 11):
        print("^^^^^^^^^^^^^^")
        print(seed)
        args.seed = seed

        ########################################################  SLDAC
        #Lv = 0
        #args.hard_flag = 1
        #args.shape_flag = 1
        #name1 = "SLDAC_Lv" + str(Lv) + "_hard" + str(args.hard_flag) + "_shape" + str(args.shape_flag)
        #print(name1)
        #SLDAC_reward, SLDAC_cost = SLDAC_main(args, Lv)
        #SLDAC_reward_save1.append(SLDAC_reward)
        #SLDAC_cost_save1.append(SLDAC_cost)
        #savemat(name1 + "_reward.mat", {"array": SLDAC_reward_save1})
        #savemat(name1 + "_cost.mat", {"array": SLDAC_cost_save1})

        ########################################################  SLDAC (spare)
        Lv = 0
        args.hard_flag = 1
        args.shape_flag = 0
        name2 = "SLDAC_Lv" + str(Lv) + "_hard" + str(args.hard_flag) + "_shape" + str(args.shape_flag)
        print(name2)
        SLDAC_reward, SLDAC_cost = SLDAC_main(args, Lv)
        SLDAC_reward_save2.append(SLDAC_reward)
        SLDAC_cost_save2.append(SLDAC_cost)
        savemat(name2 + "_reward.mat", {"array": SLDAC_reward_save2})
        savemat(name2 + "_cost.mat", {"array": SLDAC_cost_save2})

        ########################################################  SLDAC (soft)
        #Lv = 0
        #args.hard_flag = 0
        #args.shape_flag = 1
        #name3 = "SLDAC_Lv" + str(Lv) + "_hard" + str(args.hard_flag) + "_shape" + str(args.shape_flag)
        #print(name3)
        #SLDAC_reward, SLDAC_cost = SLDAC_main(args, Lv)
        #SLDAC_reward_save3.append(SLDAC_reward)
        #SLDAC_cost_save3.append(SLDAC_cost)
        #savemat(name3 + "_reward.mat", {"array": SLDAC_reward_save3})
        #savemat(name3 + "_cost.mat", {"array": SLDAC_cost_save3})

        ########################################################  SLDAC (spare+soft)
        Lv = 0
        args.hard_flag = 0
        args.shape_flag = 0
        name4 = "SLDAC_Lv" + str(Lv) + "_hard" + str(args.hard_flag) + "_shape" + str(args.shape_flag)
        print(name4)
        SLDAC_reward, SLDAC_cost = SLDAC_main(args, Lv)
        SLDAC_reward_save4.append(SLDAC_reward)
        SLDAC_cost_save4.append(SLDAC_cost)
        savemat(name4 + "_reward.mat", {"array": SLDAC_reward_save4})
        savemat(name4 + "_cost.mat", {"array": SLDAC_cost_save4})

        ########################################################  SCAOPO
        name5="SCAOPO_Lv"+str(Lv)
        print(name5)
        SCAOPO_reward, SCAOPO_cost = SCAOPO_main(args,Lv)
        SCAOPO_reward_save.append(SCAOPO_reward)
        SCAOPO_cost_save.append(SCAOPO_cost)
        savemat(name5 + "_reward.mat", {"array": SCAOPO_reward_save})
        savemat(name5 + "_cost.mat", {"array": SCAOPO_cost_save})

    Lv = 0
    name6 = "CPO_Lv0_bachsize200"
    name7 = "PPO_Lv0_bachsize200"
    ####################################################### plot
    episode = 195
    interval = 1
    x = []
    constr_limit = []
    for jj in range(int(episode / interval)):
        x.append(jj)
        constr_limit.append(0.2)


    #SLDAC_Lv0_reward = loadmat(name1+"_reward.mat")["array"]
    #SLDAC_Lv0_reward = np.mean(SLDAC_Lv0_reward, axis=0)
    #SLDAC_Lv0_reward = SLDAC_Lv0_reward[0:episode][::interval]
    SLDAC_spare_Lv0_reward = loadmat(name2+"_reward.mat")["array"]
    SLDAC_spare_Lv0_reward = np.mean(SLDAC_spare_Lv0_reward, axis=0)
    SLDAC_spare_Lv0_reward = SLDAC_spare_Lv0_reward[0:episode][::interval]
    #SLDAC_soft_Lv0_reward = loadmat(name3+"_reward.mat")["array"]
    #SLDAC_soft_Lv0_reward = np.mean(SLDAC_soft_Lv0_reward, axis=0)
    #SLDAC_soft_Lv0_reward = SLDAC_soft_Lv0_reward[0:episode][::interval]
    SLDAC_spare_soft_Lv0_reward = loadmat(name4+"_reward.mat")["array"]
    SLDAC_spare_soft_Lv0_reward = np.mean(SLDAC_spare_soft_Lv0_reward, axis=0)
    SLDAC_spare_soft_Lv0_reward = SLDAC_spare_soft_Lv0_reward[0:episode][::interval]
    SCAOPO_Lv0_reward = loadmat(name5+"_reward.mat")["array"]
    SCAOPO_Lv0_reward = np.mean(SCAOPO_Lv0_reward, axis=0)
    SCAOPO_Lv0_reward = SCAOPO_Lv0_reward[0:episode][::interval]
    PPO_Lv0_reward = loadmat(name6+"_reward.mat")["array"]
    PPO_Lv0_reward = np.mean(PPO_Lv0_reward, axis=0)
    PPO_Lv0_reward = PPO_Lv0_reward[0:episode][::interval]
    CPO_Lv0_reward = loadmat(name7+"_reward.mat")["array"]
    CPO_Lv0_reward = np.mean(CPO_Lv0_reward, axis=0)
    CPO_Lv0_reward = CPO_Lv0_reward[0:episode][::interval]


    alpha_deg = 50
    #nihe1 = np.polyfit(x, SLDAC_Lv0_reward, deg=alpha_deg)
    #SLDAC_Lv0_reward = np.polyval(nihe1, x)
    nihe2 = np.polyfit(x, SLDAC_spare_Lv0_reward, deg=alpha_deg)
    SLDAC_spare_Lv0_reward = np.polyval(nihe2, x)
    #nihe3 = np.polyfit(x, SLDAC_soft_Lv0_reward, deg=alpha_deg)
    #SLDAC_soft_Lv0_reward = np.polyval(nihe3, x)
    nihe4 = np.polyfit(x, SLDAC_spare_soft_Lv0_reward, deg=alpha_deg)
    SLDAC_spare_soft_Lv0_reward = np.polyval(nihe4, x)
    nihe5 = np.polyfit(x, SCAOPO_Lv0_reward, deg=alpha_deg)
    SCAOPO_Lv0_reward = np.polyval(nihe5, x)
    nihe6 = np.polyfit(x, PPO_Lv0_reward, deg=alpha_deg)
    PPO_Lv0_reward = np.polyval(nihe6, x)
    nihe7 = np.polyfit(x, CPO_Lv0_reward, deg=alpha_deg)
    CPO_Lv0_reward = np.polyval(nihe7, x)
    plt.figure(figsize=(9, 6.5))
    plt.plot(x, CPO_Lv0_reward, color='#556B2F', linewidth=3, linestyle='--', marker='*', markersize=1, label='CPO')
    plt.plot(x, PPO_Lv0_reward, color='#FF9900', linewidth=3, linestyle='-', marker='.', markersize=1, label='PPO-Lag')
    plt.plot(x, SCAOPO_Lv0_reward, color='m', linewidth=3, linestyle='-', label='SCAOPO')
    plt.plot(x, SLDAC_spare_Lv0_reward, color='blue', linewidth=3, linestyle='-', label='CDAC-OPS')
    plt.plot(x, SLDAC_spare_soft_Lv0_reward, color='blue', linewidth=3.5, linestyle='--', label='CDAC-OPS (soft)')
    plt.margins(x=0)
    plt.ylim(1.5, 3.5)
    plt.xlabel("iteration")
    my_x_ticks_1 = np.arange(0, int(episode/interval), 20)
    my_x_ticks_2 = np.arange(0, update_time_per_episode*episode, update_time_per_episode*interval*20)
    plt.xticks(my_x_ticks_1, my_x_ticks_2)
    plt.ylabel('Hard-delay constrained effective throughout')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig("URLLC_reward.pdf")
    plt.show()

    #SLDAC_Lv0_cost = loadmat(name1+"_cost.mat")["array"]
    #SLDAC_Lv0_cost = np.mean(SLDAC_Lv0_cost,axis=0)
    #SLDAC_Lv0_cost = SLDAC_Lv0_cost[0:episode][::interval]
    SLDAC_spare_Lv0_cost = loadmat(name2+"_cost.mat")["array"]
    SLDAC_spare_Lv0_cost = np.mean(SLDAC_spare_Lv0_cost,axis=0)
    SLDAC_spare_Lv0_cost = SLDAC_spare_Lv0_cost[0:episode][::interval]
    #SLDAC_soft_Lv0_cost = loadmat(name3+"_cost.mat")["array"]
    #SLDAC_soft_Lv0_cost = np.mean(SLDAC_soft_Lv0_cost,axis=0)
    #SLDAC_soft_Lv0_cost = SLDAC_soft_Lv0_cost[0:episode][::interval]
    SLDAC_spare_soft_Lv0_cost = loadmat(name4+"_cost.mat")["array"]
    SLDAC_spare_soft_Lv0_cost = np.mean(SLDAC_spare_soft_Lv0_cost,axis=0)
    SLDAC_spare_soft_Lv0_cost = SLDAC_spare_soft_Lv0_cost[0:episode][::interval]
    SCAOPO_Lv0_cost = loadmat(name5+"_cost.mat")["array"]
    SCAOPO_Lv0_cost = np.mean(SCAOPO_Lv0_cost,axis=0)
    SCAOPO_Lv0_cost = SCAOPO_Lv0_cost[0:episode][::interval]
    CPO_Lv0_cost = loadmat(name6+"_cost.mat")["array"]
    CPO_Lv0_cost = np.mean(CPO_Lv0_cost,axis=0)
    CPO_Lv0_cost = CPO_Lv0_cost[0:episode][::interval]
    PPO_Lv0_cost = loadmat(name7+"_cost.mat")["array"]
    PPO_Lv0_cost = np.mean(PPO_Lv0_cost,axis=0)
    PPO_Lv0_cost = PPO_Lv0_cost[0:episode][::interval]
    plt.figure(figsize=(9, 6.5))
    plt.plot(x, CPO_Lv0_cost, color='#556B2F', linewidth=3, linestyle='--', marker='*', markersize=1, label='CPO')
    plt.plot(x, PPO_Lv0_cost, color='#FF9900', linewidth=3, linestyle='-', marker='.', markersize=1, label='PPO-Lag')
    plt.plot(x, SCAOPO_Lv0_cost, color='m', linewidth=3, linestyle='-', label='SCAOPO')
    plt.plot(x, SLDAC_spare_Lv0_cost, color='blue', linewidth=3, linestyle='-', label='CDAC-OPS')
    plt.plot(x, SLDAC_spare_soft_Lv0_cost, color='blue', linewidth=3.5, linestyle='--', label='CDAC-OPS (soft)')
    plt.plot(x, constr_limit, color='black', linewidth=2, linestyle=':', label='constraint limit')
    plt.margins(x=0)
    plt.ylim(0, 1)
    plt.xlabel("iteration")
    my_x_ticks_1 = np.arange(0, int(episode/interval), 20)
    my_x_ticks_2 = np.arange(0, update_time_per_episode*episode, update_time_per_episode*interval*20)
    plt.xticks(my_x_ticks_1, my_x_ticks_2)
    plt.ylabel('Dropout Rate')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig("URLLC_cost.pdf")
    plt.show()

alpha_pow = 0.6
beta_pow = 0.7
gamma_pow = 0.3
gamma_pow_reward = gamma_pow
gamma_pow_cost = gamma_pow
tau_reward = 1
tau_cost = 1

num_new_data = 200
T = int(num_new_data/2)
grad_T = T*2
window = 30000
episode = 200
update_time_per_episode = 10
num_update_time = episode*update_time_per_episode
Q_update_time = 20
MAX_STEPS = 2*T + num_update_time*num_new_data

hard_flag = 1
shape_flag = 1

parser = argparse.ArgumentParser()
parser.add_argument('--hard_flag', type=int, default=hard_flag)
parser.add_argument('--shape_flag', type=int, default=shape_flag)
parser.add_argument('--T', type=int, default=T)
parser.add_argument('--grad_T', type=int, default=grad_T)
parser.add_argument('--window', type=int, default=window)
parser.add_argument('--num_new_data', type=int, default=num_new_data)
parser.add_argument('--episode', type=int, default=episode)
parser.add_argument('--update_time_per_episode', type=int, default=update_time_per_episode)
parser.add_argument('--num_update_time', type=int, default=num_update_time)
parser.add_argument('--Q_update_time', type=int, default=Q_update_time)
parser.add_argument('--MAX_STEPS', type=int, default=MAX_STEPS)
parser.add_argument('--alpha_pow', type=float, default=alpha_pow)
parser.add_argument('--beta_pow', type=float, default=beta_pow)
parser.add_argument('--gamma_pow_reward', type=float, default=gamma_pow_reward)
parser.add_argument('--gamma_pow_cost', type=float, default=gamma_pow_cost)
parser.add_argument('--tau_reward', type=float, default=tau_reward)
parser.add_argument('--tau_cost', type=float, default=tau_cost)
args = parser.parse_args()

main(args)