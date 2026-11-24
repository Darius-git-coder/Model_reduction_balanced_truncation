
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import helpers
    from scipy.sparse import csr_matrix, block_diag

    # Settings
    l = 9 # -> check also larger values than 9, e.g., 25 (takes longer)
    d_fine = 2*l-1 # fine grid for finite difference discretization
    d_coarse = l-1 # coarse grid with only every other grid point

    n_fine = d_fine**2 # number of inner grid points (fine)
    n_coarse = d_coarse**2 # number of inner grid points (coarse)
    tf = 1 # final time for simulation over [0,t_f]
    nt = 1000 # number of time steps used for interval (0,t_f]
    hinf_sampling_points = 1000 # sampling points used for approximate H-infinity computation
    trunctol = 1e-14 # tolerance used for computing low-rank Cholesky-like factorizations of P,Q
    I_fine = np.eye(d_fine+1)
    time_interval = np.linspace(0, tf, nt+1)
    u = lambda t: np.sin(2*np.pi*t)*10 # input signal for test purposes
    x0_fine = np.zeros((n_fine,1)) # zero initial conditions
    x0_coarse = np.zeros((n_coarse,1)) # zero initial conditions

    A_fine,B_fine,C1_fine,X_fine,Y_fine = helpers.fdm_2d_poisson(d_fine) # 2D FDM for Laplace (fine)
    A_coarse,B_coarse,C1_coarse,X_coarse,Y_coarse = helpers.fdm_2d_poisson(d_coarse) # 2D FDM for Laplace (coarse)

    C2_fine = np.eye(n_fine) # construct averaging operator which restricts from fine to coarse grid
    C2_fine_to_coarse = 1/4*sum(
            np.kron(I_fine[i::2,:][:-1,:-1], I_fine[j::2,:][:-1,:-1])
        for i in range(2) for j in range(2)
    )
    C2_coarse = np.eye(n_coarse)

    # case C1 = average value over grid (single output)
    A_bal1,B_bal1,C_bal1,Sigma_bal1 = helpers.balancing(A_fine,B_fine,C1_fine,trunctol) # compute numerically minimal balanced realization
    rmax_1 = A_bal1.shape[0]-1 # set upper limit for ROM dimensions (one less than numerical McMillan degree)
    hinf_err_bt1 = np.zeros((rmax_1,1))
    err_bnd1 = np.zeros((rmax_1,1))
    for r in range(rmax_1):
        # setting up the error system
        Ae = block_diag([A_fine, csr_matrix(A_bal1[:r,:r])])
        Be = np.vstack([B_fine, B_bal1[:r,:]])
        Ce = np.hstack([C1_fine, -C_bal1[:,:r]])
        # compute (approximate) H-infinity error
        hinf_err_bt1[r] = helpers.hinf_norm(Ae,Be,Ce,hinf_sampling_points)
        # compute a priori error bound for balanced truncation
        err_bnd1[r] = 2 * np.sum(Sigma_bal1[r:])

    # case C2 = identity (full state observation)
    A_bal2,B_bal2,C_bal2,Sigma_bal2 = helpers.balancing(A_fine,B_fine,C2_fine,trunctol) # compute numerically minimal balanced realization
    rmax_2 = A_bal2.shape[0]-1 # set upper limit for ROM dimensions (one less than numerical McMillan degree)
    hinf_err_bt2 = np.zeros((rmax_2,1))
    err_bnd2 = np.zeros((rmax_2,1))
    for r in range(rmax_2):
        # setting up the error system
        Ae = block_diag([A_fine, csr_matrix(A_bal2[:r,:r])])
        Be = np.vstack([B_fine, B_bal2[:r,:]])
        Ce = np.hstack([C2_fine, -C_bal2[:,:r]])
        # compute (approximate) H-infinity error
        hinf_err_bt2[r] = helpers.hinf_norm(Ae,Be,Ce,hinf_sampling_points)
        # compute a priori error bound for balanced truncation
        err_bnd2[r] = 2 * np.sum(Sigma_bal2[r:])

    r = 4 # comparison of r-dimensional BT-ROM and a FDM method with coarser step size
    # solving the LTI systems (approximately) with an implicit Euler method
    x_fine = helpers.implicit_euler(A_fine, B_fine, tf, u, x0_fine, nt)
    x_coarse = helpers.implicit_euler(A_coarse, B_coarse, tf, u, x0_coarse, nt)
    x1_bt = helpers.implicit_euler(A_bal1[:r,:r], B_bal1[:r,:], tf, u, np.zeros((r,1)), nt)
    x2_bt = helpers.implicit_euler(A_bal2[:r,:r], B_bal2[:r,:], tf, u, np.zeros((r,1)), nt)
    # compute all output trajectories (for two different C matrices)
    y1_fine = C1_fine @ x_fine
    y2_fine = C2_fine @ x_fine
    y_fine_to_coarse = C2_fine_to_coarse @ x_fine
    y1_coarse = C1_coarse @ x_coarse
    y2_coarse = C2_coarse @ x_coarse
    y1_bt = C_bal1[:,:r] @ x1_bt
    y2_bt = C_bal2[:,:r] @ x2_bt
    # compute relative errors for case C1
    E1_rel_bt = np.abs(y1_fine-y1_bt)/np.abs(y1_fine)
    E1_rel_coarse = np.abs(y1_fine-y1_coarse)/np.abs(y1_fine)

    # compute surfaces and relative errors for case C2
    Z_fine = np.reshape(y2_fine[:, -1], (d_fine, d_fine)).T
    Z_bt = np.reshape(y2_bt[:,-1], (d_fine, d_fine)).T
    Z_fine_to_coarse = np.reshape(y_fine_to_coarse[:, -1], (d_coarse, d_coarse)).T
    Z_coarse = np.reshape(y2_coarse[:, -1], (d_coarse, d_coarse)).T
    E2_rel_bt = np.log(np.abs(Z_fine-Z_bt)/np.abs(Z_fine))
    E2_rel_coarse = np.log(np.abs(Z_fine_to_coarse-Z_coarse)/np.abs(Z_fine_to_coarse))


    ### from here onwards, only plotting of different kind
    ## C_1 = avg
    helpers.mpl_settings(fontsize=14, figsize=(16.5,4)) #,backend='qt5agg',dpi=120)
    fig1, axs1 = plt.subplots(1, 3)

    axs1[0].semilogy(np.arange(1, rmax_1+1), hinf_err_bt1, 'ob-',label='$\|G-G_r\|_{\mathcal{H}_\infty}$')
    axs1[0].semilogy(np.arange(1, rmax_1+1), err_bnd1, 'xk--',label='$2 \sum_{k=r+1}^n \sigma_k$')
    axs1[0].set_xlabel('Reduced-order dimension')
    axs1[0].set_title('$\mathcal{H}_\infty$-error vs BT error bound')
    axs1[0].legend()

    axs1[1].plot(time_interval[::10], y1_fine[0,::10],'k-',label='$y_{1,f}(t)$ (FDM fine)', alpha=0.5)
    axs1[1].plot(time_interval[::10], y1_coarse[0,::10], 'or',label='$y_{1,c}(t)$ (FDM coarse)', alpha=0.5)
    axs1[1].plot(time_interval[::10], y1_bt[0,::10], '*b',label='$y_{1,BT}(t)$ (BT ROM)', alpha=0.5)
    axs1[1].set_xlabel('Time $t$')
    axs1[1].set_title('Output signals for $u(t)=10\sin(2\pi t)$')
    axs1[1].legend()

    axs1[2].semilogy(time_interval[::10], E1_rel_coarse[0,::10], 'or',label='FDM coarse')
    axs1[2].semilogy(time_interval[::10], E1_rel_bt[0,::10], '*b',label='BT ROM')
    axs1[2].set_xlabel('Time $t$')
    axs1[2].set_title('Relative output error for $u(t)=10\sin(2\pi t)$')
    axs1[2].legend()

    fig1.suptitle('Averaged state observation')
    fig1.tight_layout()
    fig1.savefig('c1.png')
    fig1.show()

    ## C_2 = id
    helpers.mpl_settings(fontsize=14, figsize=(22,8)) # bigger figure size needed
    fig2 = plt.figure()
    gs2 = fig2.add_gridspec(nrows=2,ncols=4) # or ncols = 4

    ax_error = fig2.add_subplot(gs2[0,3]) # or gs2[:,3:]
    ax_error.semilogy(np.arange(1, rmax_2+1), hinf_err_bt2, 'ob-',label='$\|G-G_r\|_{\mathcal{H}_\infty}$')
    ax_error.semilogy(np.arange(1, rmax_2+1), err_bnd2, 'xk--',label='$2 \sum_{k=r+1}^n \sigma_k$')
    ax_error.set_xlabel('Reduced-order dimension')
    ax_error.set_title('$\mathcal{H}_\infty$-error vs BT error bound')
    ax_error.legend()

    ax_orig = fig2.add_subplot(gs2[0,0], projection='3d')
    ax_orig.plot_surface(X_fine, Y_fine, Z_fine, rstride=1, cstride=1, linestyle='None',cmap='jet')
    ax_orig.set_xlabel('$x_1$')
    ax_orig.set_ylabel('$x_2$')
    ax_orig.view_init(elev=30, azim=45)
    ax_orig.set_xlim([X_fine.min(), X_fine.max()])
    ax_orig.set_ylim([Y_fine.min(), Y_fine.max()])
    ax_orig.set_zlim([Z_fine.min(), Z_fine.max()])
    ax_orig.set_title('FDM (fine) solution $y_2(t_f=1,x_1,x_2)$')

    ax_bt = fig2.add_subplot(gs2[0,1], projection='3d')
    ax_bt.plot_surface(X_fine, Y_fine, Z_bt, rstride=1, cstride=1, linestyle='None',cmap='jet')
    ax_bt.set_xlabel('$x_1$')
    ax_bt.set_ylabel('$x_2$')
    ax_bt.view_init(elev=30, azim=45)
    ax_bt.set_xlim([X_fine.min(), X_fine.max()])
    ax_bt.set_ylim([Y_fine.min(), Y_fine.max()])
    ax_bt.set_zlim([Z_fine.min(), Z_fine.max()])
    ax_bt.set_title('BT solution $y_2(t_f=1,x_1,x_2)$')

    ax_bt_error = fig2.add_subplot(gs2[0,2], projection='3d')
    ax_bt_error.plot_surface(X_fine, Y_fine, E2_rel_bt, rstride=1, cstride=1, linestyle='None',cmap='jet')
    ax_bt_error.set_xlabel('$x_1$')
    ax_bt_error.set_ylabel('$x_2$')
    ax_bt_error.set_zlabel('$\\log(\\text{error})$')
    ax_bt_error.view_init(elev=30, azim=45)
    ax_bt_error.set_xlim([X_fine.min(), X_fine.max()])
    ax_bt_error.set_ylim([Y_fine.min(), Y_fine.max()])
    ax_bt_error.set_zlim([E2_rel_bt.min(), E2_rel_bt.max()])
    ax_bt_error.set_title('Pointwise relative error - FDM (fine) vs BT')

    ax_fine_on_coarse = fig2.add_subplot(gs2[1,0], projection='3d')
    ax_fine_on_coarse.plot_surface(X_coarse, Y_coarse, Z_fine_to_coarse, rstride=1, cstride=1, linestyle='None',cmap='jet')
    ax_fine_on_coarse.set_xlabel('$x_1$')
    ax_fine_on_coarse.set_ylabel('$x_2$')
    ax_fine_on_coarse.view_init(elev=30, azim=45)
    ax_fine_on_coarse.set_xlim([X_fine.min(), X_fine.max()])
    ax_fine_on_coarse.set_ylim([Y_fine.min(), Y_fine.max()])
    ax_fine_on_coarse.set_zlim([Z_fine.min(), Z_fine.max()])
    ax_fine_on_coarse.set_title('FDM (fine on coarse grid) solution $y_2(t_f=1,x_1,x_2)$')

    ax_coarse = fig2.add_subplot(gs2[1,1], projection='3d')
    ax_coarse.plot_surface(X_coarse, Y_coarse, Z_coarse, rstride=1, cstride=1, linestyle='None',cmap='jet')
    ax_coarse.set_xlabel('$x_1$')
    ax_coarse.set_ylabel('$x_2$')
    ax_coarse.view_init(elev=30, azim=45)
    ax_coarse.set_xlim([X_fine.min(), X_fine.max()])
    ax_coarse.set_ylim([Y_fine.min(), Y_fine.max()])
    ax_coarse.set_zlim([Z_fine.min(), Z_fine.max()])
    ax_coarse.set_title('FDM (coarse) solution $y_2(t_f=1,x_1,x_2)$')

    ax_fine_coarse_error = fig2.add_subplot(gs2[1,2], projection='3d')
    ax_fine_coarse_error.plot_surface(X_coarse, Y_coarse, E2_rel_coarse, rstride=1, cstride=1, linestyle='None',cmap='jet')
    ax_fine_coarse_error.set_xlabel('$x_1$')
    ax_fine_coarse_error.set_ylabel('$x_2$')
    ax_fine_coarse_error.set_zlabel('$\\log(\\text{error})$')
    ax_fine_coarse_error.view_init(elev=30, azim=45)
    ax_fine_coarse_error.set_xlim([X_fine.min(), X_fine.max()])
    ax_fine_coarse_error.set_ylim([Y_fine.min(), Y_fine.max()])
    ax_fine_coarse_error.set_zlim([E2_rel_coarse.min(), E2_rel_coarse.max()])
    ax_fine_coarse_error.set_title('Pointwise relative error - FDM (fine) vs FDM (coarse)')

    fig2.suptitle('Full state observation')
    fig2.tight_layout()
    fig2.savefig('c2.png')
    fig2.show()
