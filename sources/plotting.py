
py.close('all')
fig = py.figure(figsize=(10.0, 5.0))
ax = fig.add_subplot(111)
fig.suptitle('AAPL, March-April 2008', fontsize=20, fontweight='bold')

den = 0.008

sel_side = aapl.side==1
aapl_buy = aapl[sel_side]
py.hlines(aapl_buy.day_trade_n, aapl_buy.mm_s, aapl_buy.mm_e, linestyles='solid', lw= py.array(aapl_buy.eta/den), color='blue', alpha=0.3)

sel_side = aapl.side==-1
aapl_sell = aapl[sel_side]
py.hlines(aapl_sell.day_trade_n, aapl_sell.mm_s, aapl_sell.mm_e, linestyles='solid', lw= py.array(aapl_sell.eta/den), color='red', alpha=0.3)


ax.set_xlim([0,390])
ax.set_ylim([len(all_days),-1])
ax.set_aspect('auto')
ax.set_xlabel('Trading minute')
ax.set_ylabel('Trading day')
py.subplots_adjust(bottom=0.15)
py.savefig("../plot/aapl.pdf")


##############################


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



# plotting
x_plf = pow(10,py.linspace(-6,0,1000))

# distance between axes and ticks
py.rcParams['xtick.major.pad']='8'
py.rcParams['ytick.major.pad']='8'

# set latex font
py.rc('text', usetex=True)
py.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})

py.clf()
p_pl, = py.plot(x_plf,ff_pl(x_plf,par_pl[0],par_pl[1]), ls='--', color='Red')
p_lg, = py.plot(x_plf,ff_lg(x_plf,par_lg[0],par_lg[1]), ls='-', color='RoyalBlue')
p_points, = py.plot(df_out.pi,df_out.imp,'.', color='Black',ms=10)
py.xscale('log')
py.yscale('log')
py.xlabel('$\phi$')
py.ylabel('$\mathcal{I}_{tmp}(\Omega=\{ \phi \})$')
py.grid()
py.axis([0.00001,1,0.0001,0.1])
leg_1 = '$\hat{Y} = $' + str("%.4f" % round(par_pl[0],4)) + '$\pm$' + str("%.4f" % round(vv_pl[0][0],4)) + ' $\hat{\delta} = $' + str("%.4f" % round(par_pl[1],4)) + '$\pm$' + str("%.4f" % round(vv_pl[1][1],4)) + ' $E_{RMS} = $' + str("%.4f" % round(py.sqrt(chi_pl/len(df_out.imp)),4))
leg_2 = '$\hat{a} = $' + str("%.3f" % round(par_lg[0],3)) + '$\pm$' + str("%.3f" % round(vv_lg[0][0],3)) + ' $\hat{b} = $' + str("%.0f" % round(par_lg[1],3)) + '$\pm$' + str("%.0f" % round(vv_lg[1][1],3)) + ' $E_{RMS} = $' + str("%.4f" % round(py.sqrt(chi_lg/len(df_out.imp)),4))
l1 = py.legend([p_pl,p_lg], ['$f(\phi) = Y\phi^{\delta}$', '$g(\phi)= a \log_{10}(1+b\phi)$'], loc=2, prop={'size':15})
l2 = py.legend([p_pl,p_lg], [leg_1 ,leg_2 ], loc=4, prop={'size':15})
py.gca().add_artist(l1)
py.subplots_adjust(bottom=0.15)
py.subplots_adjust(left=0.17)
py.savefig("../plot/imp_1d.pdf")
