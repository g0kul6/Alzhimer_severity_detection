import numpy as np
import matplotlib.pyplot as plt

acc_val=np.load("Plots/accuracy/accuracy_val.npy")
loss_val=np.load("Plots/loss/loss_val.npy")
acc_train=np.load("Plots/accuracy/accuracy_train.npy")
loss_train=np.load("Plots/loss/loss_train.npy")


plt.plot(range(500),acc_train,color ='blue', label ='train')
plt.plot(range(500),acc_val,color ='orange', label ='val')
plt.legend()
plt.show()

