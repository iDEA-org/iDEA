import numpy as np
import scipy.special as spspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import iDEA


x = np.linspace(-10, 10, 300)
v_ext = -2.0 / (abs(x) + 1.0) # 0.5 * (0.25)**2 * x**2
v_int = iDEA.interactions.softened_interaction(x) #* 1e-10
electrons = 'u'
s = iDEA.system.System(x, v_ext, v_int, electrons)


state = iDEA.methods.hartree_fock.solve(s, k=0)


n = iDEA.observables.density(s, state=state)
v_h = iDEA.observables.hartree_potential(s, n)
E_h = iDEA.observables.hartree_energy(s, n, v_h)

p = iDEA.observables.density_matrix(s, state=state)
v_x = iDEA.observables.exchange_potential(s, p)
E_x = iDEA.observables.exchange_energy(s, p, v_x)


plt.plot(s.x, n, 'k-')
plt.plot(s.x, np.diagonal(p), 'r--')
plt.plot(s.x, v_h, 'g-')
plt.show()


plt.imshow(v_x)
plt.show()


orb = state.up.orbitals[:,0]
v_x_eff = np.zeros(shape = v_h.shape[0], dtype=float)
for i in range(v_x_eff.shape[0]):  
    v_x_eff[i] = (1.0/orb[i]) * np.sum(v_x[i,:] * orb[:]) * s.dx


plt.plot(s.x, v_x_eff + v_h, 'k-')
plt.show()


print(E_h, E_x)


t = np.linspace(0, 100, 500)
v_ptrb = np.zeros(shape=t.shape+x.shape)
for j, ti in enumerate(t):
    v_ptrb[j,:] = -0.1 * x * 0.5*(spspec.erf(ti-2)+1)


evolution = iDEA.methods.non_interacting.propagate(s, state, v_ptrb, t)


n = iDEA.observables.density(s, evolution=evolution)
v_h = iDEA.observables.hartree_potential(s, n)
E_h = iDEA.observables.hartree_energy(s, n, v_h)

p = iDEA.observables.density_matrix(s, evolution=evolution)
v_x = iDEA.observables.exchange_potential(s, p)
E_x = iDEA.observables.exchange_energy(s, p, v_x)


# plt.plot(t, E_h, 'k-')
# plt.plot(t, E_x, 'r-')
plt.plot(t, E_h + E_x, 'g-')
plt.show()


orb = evolution.up.td_orbitals[:,:,0]
v_x_eff = np.zeros(shape = v_h.shape, dtype=float)
for j in range(t.shape[0]):
    for i in range(v_x_eff.shape[1]):  
        v_x_eff[j,i] = (1.0/orb[j,i]) * np.sum(v_x[j,i,:] * orb[j,:]) * s.dx
    print(np.sum(v_x_eff[j,:] + v_h[j,:]) * s.dx)


fig = plt.figure()
ax = plt.axes(xlim=(s.x[0], s.x[-1]), ylim=(-0.08, 0.08))
line1, = ax.plot([], [], lw=2, c='k')
line2, = ax.plot([], [], lw=2, c='r', ls='--')
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2, 
def animate(i):
    line1.set_data(s.x, v_x_eff[i,:] + v_h[i,:])
    line2.set_data(s.x, v_x_eff[i,:] + v_h[i,:])
    return line1, line2,
anim = FuncAnimation(fig, animate, init_func=init, frames=100, interval=10, blit=True)
plt.show()


# t = np.linspace(0, 10, 100)
# v_ptrb = np.zeros(shape=t.shape+x.shape)
# for j, ti in enumerate(t):
#     v_ptrb[j,:] = -0.1 * x * 0.5*(spspec.erf(ti-2)+1) #* np.sin(2*np.pi*0.01*ti)


# state = iDEA.methods.interacting.solve(s, k=0)
# target_n = iDEA.observables.density(s, state=state)
# s_fictious = iDEA.reverse_engineering.reverse(s, target_n, method=iDEA.methods.non_interacting, tol=1e-12)
# state_fictious = iDEA.methods.non_interacting.solve(s_fictious)

# evolution = iDEA.methods.interacting.propagate(s, state, v_ptrb, t)
# target_n = iDEA.observables.density(s, evolution=evolution)
# evolution_fictious, error = iDEA.reverse_engineering.reverse_propagation(s_fictious, state_fictious, target_n, iDEA.methods.non_interacting, v_ptrb, t)
# evolution_fictious = iDEA.methods.non_interacting.propagate(s_fictious, state_fictious, evolution_fictious.v_ptrb, t)

# for ti in range( len(t) ):
#     evolution_fictious.v_ptrb[ti,:] -= evolution_fictious.v_ptrb[ti,int(0.5*len(x))]


# plt.plot(t, error, 'k-')
# plt.show()


# n = iDEA.observables.density(s, evolution=evolution_fictious)
# v_ext = iDEA.observables.external_potential(s)
# v_h = iDEA.observables.hartree_potential(s, n)


# fig = plt.figure()
# ax = plt.axes(xlim=(s.x[0], s.x[-1]), ylim=(-3.0, 3.0))
# line1, = ax.plot([], [], lw=2, c='k')
# line2, = ax.plot([], [], lw=2, c='r', ls='--')
# line3, = ax.plot([], [], lw=2, c='c')
# line4, = ax.plot([], [], lw=2, c='g', ls='--')
# def init():
#     line1.set_data([], [])
#     line2.set_data([], [])
#     line3.set_data([], [])
#     line4.set_data([], [])
#     return line1, line2, line3, line4
# def animate(i):
#     line1.set_data(s.x, n[i,:])
#     line2.set_data(s.x, target_n[i,:])
#     line3.set_data(s.x, evolution_fictious.v_ptrb[i,:]) # s_fictious.v_ext[:] +
#     line4.set_data(s.x, v_ptrb[i,:])
#     return line1, line2, line3, line4
# anim = FuncAnimation(fig, animate, init_func=init, frames=100, interval=10, blit=True)
# plt.show()



























































# t = np.linspace(0, 100, 1000)
# v_ptrb = np.zeros(shape=t.shape+x.shape)
# for j, ti in enumerate(t):
#     v_ptrb[j,:] = -0.1 * x #* np.sin(0.5*ti)


# state = iDEA.methods.interacting.solve(s)
# n = iDEA.observables.density(s, state=state)


# s_fictious = iDEA.reverse.reverse(s, n, method=iDEA.methods.non_interacting)
# state_fictious = iDEA.methods.non_interacting.solve(s_fictious, k=0)
# n_fictious = iDEA.observables.density(s, state=state_fictious)


# v_ks = s_fictious.v_ext
# v_ext = s.v_ext
# v_h = iDEA.observables.hartree_potential(s_fictious, n_fictious)
# v_xc = v_ks - v_ext - v_h


# plt.plot(s_fictious.x, n_fictious, 'k-')
# plt.plot(s_fictious.x, v_ks, 'r-')
# plt.plot(s_fictious.x, v_ext, 'g-')
# plt.plot(s_fictious.x, v_h, 'b-')
# plt.plot(s_fictious.x, v_xc, 'c-')
# plt.show()


# evolution = iDEA.methods.interacting.propagate(s, state, v_ptrb, t)
# target_n = iDEA.observables.density(s, evolution=evolution)
# evolution_fictious, error = iDEA.reverse.reverse_propigation(s_fictious, state_fictious, target_n, method=iDEA.methods.non_interacting, restricted=False, v_ptrb=v_ptrb, t=t)


# method = iDEA.methods.hybrid
# restricted = False
# reverse = False
# compare = False
# k = 0
# time_dependence = True
# td_reverse = True
# hybrid_alpha = 0.9


# if reverse == True:
#     state = iDEA.methods.interacting.solve(s)
#     target_n = iDEA.observables.density(s, state=state)
#     s_fictious = iDEA.reverse.reverse(s, target_n, method=iDEA.methods.non_interacting, mixing=1.0)
#     v_ks = s_fictious.v_ext
#     v_ext = s.v_ext
#     v_h = iDEA.observables.hartree_potential(s, target_n)
#     v_xc = v_ks - v_ext - v_h
#     plt.plot(s.x, target_n)
#     plt.plot(s.x, v_ks)
#     plt.plot(s.x, v_ext)
#     plt.plot(s.x, v_h)
#     plt.plot(s.x, v_xc)
#     plt.show()


# if compare == False:
#     if method == iDEA.methods.interacting:
#         state = method.solve(s, k=k)
#         E = method.total_energy(s, state)
#         print(k, E)
#     elif method == iDEA.methods.hybrid:
#         state = method.solve(s, k=k, restricted=restricted, alpha=hybrid_alpha)
#         E = method.total_energy(s, state)
#         print(k, E)
#     else:
#         state = method.solve(s, k=k, restricted=restricted)
#         E = method.total_energy(s, state)
#         print(k, E)


#     n = iDEA.observables.density(s, state=state)
#     p = iDEA.observables.density_matrix(s, state=state)
#     v_ext = iDEA.observables.external_potential(s)
#     E_ext = iDEA.observables.external_energy(s, n, v_ext)
#     v_h = iDEA.observables.hartree_potential(s, n)
#     E_h = iDEA.observables.hartree_energy(s, n, v_h)
#     v_x = iDEA.observables.exchange_potential(s, p)
#     E_x = iDEA.observables.exchange_energy(s, p, v_x)


#     print(E_ext, E_h, E_x)
#     plt.imshow(p)
#     plt.show()
#     plt.imshow(v_x)
#     plt.show()
#     plt.plot(s.x, n, 'k-')
#     plt.plot(s.x, np.diagonal(p), 'r--')
#     plt.plot(s.x, v_ext, 'c-')
#     plt.plot(s.x, v_h, 'g-')
#     plt.show()


# else:
#     p = ['k-', 'r--', 'b--', 'c--', 'g--', 'm--']
#     for i, m in enumerate(iDEA.iterate_methods):
#         if m == iDEA.methods.interacting:
#             state = m.solve(s, k=k)
#         elif m == iDEA.methods.hybrid:
#             state = m.solve(s, k=k, restricted=restricted, alpha=hybrid_alpha)
#         elif m == iDEA.methods.hartree:
#             state = m.solve(s, k=k, restricted=restricted, tol=1e-5)
#         else:
#             state = m.solve(s, k=k, restricted=restricted)
#         n = iDEA.observables.density(s, state=state)
#         plt.plot(s.x, n, p[i], label=m.name)
#     plt.legend()
#     plt.show()


# if time_dependence:
#     if method == iDEA.methods.interacting:
#         evolution = method.propagate(s, state, v_ptrb, t)
#     elif method == iDEA.methods.hybrid:
#         evolution = method.propagate(s, state, v_ptrb, t, restricted=restricted, alpha=hybrid_alpha)
#     else:
#         evolution = method.propagate(s, state, v_ptrb, t, restricted=restricted)


#     n = iDEA.observables.density(s, evolution=evolution)
#     p = iDEA.observables.density_matrix(s, evolution=evolution)
#     v_ext = iDEA.observables.external_potential(s)
#     E_ext = iDEA.observables.external_energy(s, n, v_ext)
#     v_h = iDEA.observables.hartree_potential(s, n)
#     E_h = iDEA.observables.hartree_energy(s, n, v_h)
#     v_x = iDEA.observables.exchange_potential(s, p)
#     E_x = iDEA.observables.exchange_energy(s, p, v_x)

#     plt.plot(t, E_ext, label='E_ext')
#     plt.plot(t, E_h, label='E_h')
#     plt.plot(t, E_x, label='E_x')
#     plt.legend()
#     plt.show()

#     fig = plt.figure()
#     ax = plt.axes(xlim=(s.x[0], s.x[-1]), ylim=(-3.0, 3.0))
#     line1, = ax.plot([], [], lw=2, c='k')
#     line2, = ax.plot([], [], lw=2, c='r', ls='--')
#     line3, = ax.plot([], [], lw=2, c='g')
#     line4, = ax.plot([], [], lw=2, c='c')
#     def init():
#         line1.set_data([], [])
#         line2.set_data([], [])
#         line3.set_data([], [])
#         line4.set_data([], [])
#         return line1, line2, line3, line4
#     def animate(i):
#         line1.set_data(s.x, n[i,:])
#         line2.set_data(s.x, np.diagonal(p[i,:,:].real))
#         line3.set_data(s.x, v_h[i,:])
#         line4.set_data(s.x, v_ext[:] + v_ptrb[i,:])
#         return line1, line2, line3, line4
#     anim = FuncAnimation(fig, animate, init_func=init, frames=100, interval=10, blit=True)
#     plt.show()
