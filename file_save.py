from scipy.io import loadmat
from scipy.io import savemat
import numpy as np

Lv = 0
T = 150
grad_T = T
num_new_data = 50
Q_update_time = 10
name1 = "SLDAC1_Lv" + str(Lv) + "_bachsize" + str(num_new_data) + "_T" + str(T)
name2 = "SLDAC2_Lv" + str(Lv) + "_bachsize" + str(num_new_data) + "_T" + str(T)
name3 = "SLDAC3_Lv" + str(Lv) + "_bachsize" + str(num_new_data) + "_T" + str(T)
name4 = "SLDAC4_Lv" + str(Lv) + "_bachsize" + str(num_new_data) + "_T" + str(T)
file1 = loadmat(name1+"_cost.mat")["array"]
file2 = loadmat(name2+"_cost.mat")["array"]
file3 = loadmat(name3+"_cost.mat")["array"]
file4 = loadmat(name4+"_cost.mat")["array"]
name = "SLDAC_Lv" + str(Lv) + "_bachsize" + str(num_new_data) + "_T" + str(T)
file=np.zeros((10,200),dtype=float)
file[0:3,:]=file1
file[0:3,:]=file1
file[5:7,:]=file3
file[7:10,:]=file4
savemat(name + "_cost.mat", {"array": file})
a=1