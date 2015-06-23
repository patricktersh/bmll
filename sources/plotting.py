
def plot_ge():

	# distance between axes and ticks
	py.rcParams['xtick.major.pad']='8'
	py.rcParams['ytick.major.pad']='8'

	# set latex font
	py.rc('text', usetex=True)
	py.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})


	py.close('all')
	fig = py.figure(figsize=(10.0, 5.0))
	ax = fig.add_subplot(111)
	fig.suptitle('GE, Janaury-February 2007', fontsize=20, fontweight='bold')

	den = 0.008
	N_max_day = 50
	
	sel_side = (df_ge.side==1) & (df_ge.day_trade_n < N_max_day)
	df_ge_buy = df_ge[sel_side]
	py.hlines(df_ge_buy.day_trade_n, df_ge_buy.mm_s, df_ge_buy.mm_e, linestyles='solid', lw= py.array(df_ge_buy.eta/den), color='blue', alpha=0.3)

	sel_side = (df_ge.side==-1) & (df_ge.day_trade_n < N_max_day)
	df_ge_sell = df_ge[sel_side]
	py.hlines(df_ge_sell.day_trade_n, df_ge_sell.mm_s, df_ge_sell.mm_e, linestyles='solid', lw= py.array(df_ge_sell.eta/den), color='red', alpha=0.3)


	ax.set_xlim([0,390])
	ax.set_ylim([N_max_day,-1])
	ax.set_aspect('auto')
	ax.set_xlabel('Trading minute')
	ax.set_ylabel('Trading day')
	py.subplots_adjust(bottom=0.15)
	py.savefig("../plot/ge.pdf")

	
	##############################

def plot_hist():

	# distance between axes and ticks
	py.rcParams['xtick.major.pad']='8'
	py.rcParams['ytick.major.pad']='8'

	# set latex font
	py.rc('text', usetex=True)
	py.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})


	py.close('all')
	fig = py.figure()
	ax = fig.add_subplot(1, 1, 1)
	py.plot(bins_pi_cent, pdf_pi, 'o', ms=3, color='SteelBlue')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('$Q/V_D$')
	ax.set_ylabel('$p(Q/V_D)$')
	py.grid()
	py.subplots_adjust(bottom=0.15)
	py.subplots_adjust(left=0.17)
	py.savefig("../plot/stat_pi.pdf")


	##############################

def plot_tmp_imp():

	# distance between axes and ticks
	py.rcParams['xtick.major.pad']='8'
	py.rcParams['ytick.major.pad']='8'

	# set latex font
	py.rc('text', usetex=True)
	py.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})

	# plotting
	x_plf = pow(10,py.linspace(-6,0,1000))

	py.clf()
	p_pl, = py.plot(x_plf,ff_pl(x_plf,par_pl[0],par_pl[1]), ls='--', color='Red')
	p_lg, = py.plot(x_plf,ff_lg(x_plf,par_lg[0],par_lg[1]), ls='-', color='RoyalBlue')
	p_points, = py.plot(df_imp_1d.pi,df_imp_1d.imp,'.', color='Black',ms=10)
	py.xscale('log')
	py.yscale('log')
	py.xlabel('$\phi$')
	py.ylabel('$\mathcal{I}_{tmp}(\Omega=\{ \phi \})$')
	py.grid()
	py.axis([0.00001,1,0.0001,0.1])

	leg_1 = '$\hat{Y} = $' + str("%.4f" % round(par_pl[0],4)) + '$\pm$' + str("%.4f" % round(vv_pl[0][0],4)) + ' $\hat{\delta} = $' + str("%.4f" % round(par_pl[1],4)) + '$\pm$' + str("%.4f" % round(vv_pl[1][1],4)) + ' $E_{RMS} = $' + str("%.4f" % round(py.sqrt(chi_pl/len(df_imp_1d.imp)),4))
	leg_2 = '$\hat{a} = $' + str("%.3f" % round(par_lg[0],3)) + '$\pm$' + str("%.3f" % round(vv_lg[0][0],3)) + ' $\hat{b} = $' + str("%.0f" % round(par_lg[1],3)) + '$\pm$' + str("%.0f" % round(vv_lg[1][1],3)) + ' $E_{RMS} = $' + str("%.4f" % round(py.sqrt(chi_lg/len(df_imp_1d.imp)),4))
	l1 = py.legend([p_pl,p_lg], ['$f(\phi) = Y\phi^{\delta}$', '$g(\phi)= a \log_{10}(1+b\phi)$'], loc=2, prop={'size':15})
	l2 = py.legend([p_pl,p_lg], [leg_1 ,leg_2 ], loc=4, prop={'size':15})
	py.gca().add_artist(l1)
	py.subplots_adjust(bottom=0.15)
	py.subplots_adjust(left=0.17)
	py.savefig("../plot/imp_1d.pdf")


	########################
def plot_tra_imp():

	# distance between axes and ticks
	py.rcParams['xtick.major.pad']='8'
	py.rcParams['ytick.major.pad']='8'

	# set latex font
	py.rc('text', usetex=True)
	py.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})


	N_max_imp = 200

	py.close('all')
	fig = py.figure()
	ax = fig.add_subplot(1, 1, 1)
	py.plot(range(N_max_imp),tmp_imp[:N_max_imp]/tmp_count[:N_max_imp], 'o', ms=3, color='SteelBlue')
	ax.set_xlabel('$t$')
	ax.set_ylabel('$\mathcal{I}(t)$')
	py.grid()
	py.subplots_adjust(bottom=0.15)
	py.subplots_adjust(left=0.17)
	py.savefig("../plot/imp_tra.pdf")











