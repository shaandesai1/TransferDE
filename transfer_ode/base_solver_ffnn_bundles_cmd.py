if args.viz:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)
if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size
    # define coefficients as lambda functions, used for gt and wout_analytic
    # training differential equation

    #need to sample tuple of (a1,f,IC)
    # each column of Wouts defines a solution thus, each tuple defines a solution too


    f_train = [lambda t: torch.cos(t),lambda t: torch.sin(t)]
    a0_train = [lambda t: t,lambda t:t**2]
    r1 = -5.
    r2 = 5.
    true_y0 = (r2 - r1) * torch.rand(100) + r1
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    f_samples = random.choices(f_train, k=args.num_bundles)
    a0_samples = random.choices(a0_train, k=args.num_bundles)
    y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles)).reshape(1,-1)

    diffeq_init = diffeq(a0_samples,f_samples)
    gt_generator = base_diffeq(diffeq_init)
    true_y = gt_generator.get_solution(y0_samples,t.ravel()).reshape(-1,args.num_bundles)

    # use this quick test to find gt solutions and check training ICs
    # have a solution (don't blow up for dopri5 integrator)
    # true_y = gt_generator.get_solution(true_y0.reshape(-1, 1), t.ravel())

    # instantiate wout with coefficients

    print("NDIMZ", NDIMZ)
    func = ODEFunc(hidden_dim=NDIMZ, output_dim=args.num_bundles)

    optimizer = optim.Adam(func.parameters(), lr=1e-3, weight_decay=1e-6)

    loss_collector = []

    if not args.evaluate_only:

        for itr in range(1, args.niters + 1):
            func.train()

            # add t0 to training times, including randomly generated ts
            t0 = torch.tensor([[0.]])
            t0.requires_grad = True
            tv = args.tmax * torch.rand(int(args.tmax / args.dt)).reshape(-1, 1)
            tv.requires_grad = True
            tv = torch.cat([t0, tv], 0)
            optimizer.zero_grad()

            # compute hwout,hdotwout
            pred_y = func(tv)
            pred_ydot = diff(pred_y, tv)

            # enforce diffeq
            loss_diffeq = pred_ydot - get_udot(tv,pred_y,a0_samples,f_samples)
            # loss_diffeq = (a1(tv.detach()).reshape(-1, 1)) * pred_ydot + (a0(tv.detach()).reshape(-1, 1)) * pred_y - f(
            #     tv.detach()).reshape(-1, 1)

            # enforce initial conditions
            loss_ics = pred_y[0, :].ravel() - y0_samples.ravel()

            loss = torch.mean(torch.square(loss_diffeq)) + torch.mean(torch.square(loss_ics))
            loss.backward()
            optimizer.step()
            loss_collector.append(torch.square(loss_diffeq).mean().item())
            if itr % args.test_freq == 0:
                func.eval()
                pred_y = func(t).detach()
                pred_y = pred_y.reshape(-1, args.num_bundles)
                visualize(true_y.detach(), pred_y.detach(), loss_collector)
                ii += 1

        torch.save(func.state_dict(), 'func_ffnn_bundles')

    # with torch.no_grad():

    f_test = [lambda t: torch.sin(t)]
    a0_test = [lambda t: t**3]
    r1 = -15.
    r2 = 15.
    true_y0 = (r2 - r1) * torch.rand(100) + r1
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    f_samples = random.choices(f_test, k=args.num_bundles_test)
    a0_samples = random.choices(a0_test, k=args.num_bundles_test)
    y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles_test)).reshape(1, -1)

    # print(y0_samples.shape)
    diffeq_init = diffeq(a0_samples, f_samples)
    gt_generator = base_diffeq(diffeq_init)


    func.load_state_dict(torch.load('func_ffnn_bundles'))
    func.eval()

    h = func.h(t)
    hd = diff(h, t)
    h = h.detach()
    hd = hd.detach()

    gz_np = h.detach().numpy()
    T = np.linspace(0, 1, len(gz_np)) ** 2
    new_hiddens = scaler.fit_transform(gz_np)
    pca = PCA(n_components=3)
    pca_comps = pca.fit_transform(new_hiddens)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if pca_comps.shape[1] >= 2:
        s = 10  # Segment length
        for i in range(0, len(gz_np) - s, s):
            ax.plot3D(pca_comps[i:i + s + 1, 0], pca_comps[i:i + s + 1, 1], pca_comps[i:i + s + 1, 2],
                      color=(0.1, 0.8, T[i]))
            plt.xlabel('comp1')
            plt.ylabel('comp2')


    s1 = time.time()
    wout = get_wout(h, hd, y0_samples, t.detach(),a0_samples[0],f_samples)
    pred_y = h @ wout
    s2 = time.time()
    print(f'all_ics:{s2 - s1}')

    s1 = time.time()
    true_ys = (gt_generator.get_solution(y0_samples, t.ravel())).reshape(-1, args.num_bundles_test)
    s2 = time.time()
    print(f'gt_ics:{s2 - s1}')

    print(true_ys.shape,pred_y.shape)

    # s1 = time.time()
    # true_y = estim_generator.get_solution(ics.reshape(-1, 1), t.ravel())
    # estim_ys = true_y.reshape(len(pred_y), ics.shape[1])
    # s2 = time.time()
    # print(f'estim_ics:{s2 - s1}')

    # print(f'prediction_accuracy:{((pred_y - true_ys) ** 2).mean()} pm {((pred_y - true_ys) ** 2).std()}')
    # print(f'estim_accuracy:{((estim_ys - true_ys) ** 2).mean()} pm {((estim_ys - true_ys) ** 2).std()}')

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # print(true_ys[0,:])
    for i in range(0, args.num_bundles_test, 50):
        ax[0].plot(t.detach().cpu().numpy(), true_ys.cpu().numpy()[:, i], c='blue', linestyle='dashed')
        ax[0].plot(t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i], c='orange')
        # plt.draw()

    ax[1].plot(t.detach().cpu().numpy(), ((true_ys - pred_y) ** 2).mean(1).cpu().numpy(), c='green')
    ax[1].set_xlabel('Time (s)')
    plt.legend()
    plt.show()
