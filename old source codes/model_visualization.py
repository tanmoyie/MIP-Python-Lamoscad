"""Talk about this Python script
File Name:
Outline:
1. surface plot
2. grblogtool
3.

"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


#%% Draw surface plot


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = [2, 1, 5]
Y = [1, 3, 2]
X, Y = np.meshgrid(X, Y)
Z = np.sin([0.3, 1, 9.2])

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#%%
summary = glt.parse("Outputs/model.log").summary()


#%%
print('Print Xn')
model.setParam(GRB.Param.SolutionNumber, 34)
[select[s, r].Xn for s in Stations for r in ResourcesD if select[s, r].Xn > 0.5]
[deploy[s, o, r].Xn for s in Stations for o in OilSpills for r in ResourcesD]
model.PoolObjVal