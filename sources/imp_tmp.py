import pylab as pl
import scipy.optimize as optimize
import pandas as pd
from datetime import datetime

#import functions as fn

plotting_tmp = True # set this variable to True if you want to produce the plots

###################################################################################################################
###################################################################################################################

										###################################
										####      Some functions       ####
										###################################



def extract_day( str_in ):
    return str_in[0:10]

def extract_min( str_in ):
    h = int(str_in[11:13])
    m = int(str_in[14:16])
    return h*60+m-569

def find_pos(  list_1, list_2 ):
    return map(lambda x:list_2.index(x), list_1)

def fit_nonlin_1d_2p(ff_in, df_in, a_r, b_r):
    par = []
    vv = []
    chi = []
    for a_s in pl.linspace(a_r[0],a_r[1],10):
        for b_s in pl.linspace(b_r[0],b_r[1],10):
            p0 = [a_s, b_s] # Initial guess for the parameters
            fit_in = optimize.curve_fit(ff_in,df_in.pi,df_in.imp,p0,pl.array(df_in.stdd)/pl.array(pl.sqrt(df_in.nn)))
            par.append(fit_in[0])
            vv.append(fit_in[1])
            imp_pred = ff_in(df_in.pi,fit_in[0][0],fit_in[0][1])
            chi.append( sum(pow(imp_pred-df_in.imp,2)) )

    ind_min = chi.index(min(chi))
    chi_opt = chi[ind_min]
    par_opt = par[ind_min]
    vv_opt = vv[ind_min]
    return par_opt, vv_opt, chi_opt

parse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')



###################################
####  Some plotting functions  ####
###################################



def plot_ge( name_plot ):

	# distance between axes and ticks
	pl.rcParams['xtick.major.pad']='8'
	pl.rcParams['ytick.major.pad']='8'

	# set latex font
	pl.rc('text', usetex=True)
	pl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})


	pl.close('all')
	fig = pl.figure(figsize=(10.0, 5.0))
	ax = fig.add_subplot(111)
	fig.suptitle('GE, Janaury-February 2007', fontsize=20, fontweight='bold')

	den = 0.008
	N_max_day = 50
	
	sel_side = (df_ge.side==1) & (df_ge.day_trade_n < N_max_day)
	df_ge_buy = df_ge[sel_side]
	pl.hlines(df_ge_buy.day_trade_n, df_ge_buy.mm_s, df_ge_buy.mm_e, linestyles='solid', lw= pl.array(df_ge_buy.eta/den), color='blue', alpha=0.3)

	sel_side = (df_ge.side==-1) & (df_ge.day_trade_n < N_max_day)
	df_ge_sell = df_ge[sel_side]
	pl.hlines(df_ge_sell.day_trade_n, df_ge_sell.mm_s, df_ge_sell.mm_e, linestyles='solid', lw= pl.array(df_ge_sell.eta/den), color='red', alpha=0.3)


	ax.set_xlim([0,390])
	ax.set_ylim([N_max_day,-1])
	ax.set_aspect('auto')
	ax.set_xlabel('Trading minute')
	ax.set_ylabel('Trading day')
	pl.subplots_adjust(bottom=0.15)
	pl.savefig("../plot/" + name_plot + ".pdf")

	
	##############################


def plot_hist( name_plot ):

	# distance between axes and ticks
	pl.rcParams['xtick.major.pad']='8'
	pl.rcParams['ytick.major.pad']='8'

	# set latex font
	pl.rc('text', usetex=True)
	pl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})


	pl.close('all')
	fig = pl.figure()
	ax = fig.add_subplot(1, 1, 1)
	pl.plot(bins_pi_cent, pdf_pi, 'o', ms=3, color='SteelBlue')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('$Q/V_D$')
	ax.set_ylabel('$p(Q/V_D)$')
	pl.grid()
	pl.subplots_adjust(bottom=0.15)
	pl.subplots_adjust(left=0.17)
	pl.savefig("../plot/" + name_plot + ".pdf")


	##############################


def plot_tmp_imp( name_plot ):

	# distance between axes and ticks
	pl.rcParams['xtick.major.pad']='8'
	pl.rcParams['ytick.major.pad']='8'

	# set latex font
	pl.rc('text', usetex=True)
	pl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})

	# plotting
	x_plf = pow(10,pl.linspace(-6,0,1000))

	pl.clf()
	p_pl, = pl.plot(x_plf,ff_pl(x_plf,par_pl[0],par_pl[1]), ls='--', color='Red')
	p_lg, = pl.plot(x_plf,ff_lg(x_plf,par_lg[0],par_lg[1]), ls='-', color='RoyalBlue')
	p_points, = pl.plot(df_imp_1d.pi,df_imp_1d.imp,'.', color='Black',ms=10)
	pl.xscale('log')
	pl.yscale('log')
	pl.xlabel('$\phi$')
	pl.ylabel('$\mathcal{I}_{tmp}(\Omega=\{ \phi \})$')
	pl.grid()
	pl.axis([0.00001,1,0.0001,0.1])

	leg_1 = '$\hat{Y} = $' + str("%.4f" % round(par_pl[0],4)) + '$\pm$' + str("%.4f" % round(vv_pl[0][0],4)) + ' $\hat{\delta} = $' + str("%.4f" % round(par_pl[1],4)) + '$\pm$' + str("%.4f" % round(vv_pl[1][1],4)) + ' $E_{RMS} = $' + str("%.4f" % round(pl.sqrt(chi_pl/len(df_imp_1d.imp)),4))
	leg_2 = '$\hat{a} = $' + str("%.3f" % round(par_lg[0],3)) + '$\pm$' + str("%.3f" % round(vv_lg[0][0],3)) + ' $\hat{b} = $' + str("%.0f" % round(par_lg[1],3)) + '$\pm$' + str("%.0f" % round(vv_lg[1][1],3)) + ' $E_{RMS} = $' + str("%.4f" % round(pl.sqrt(chi_lg/len(df_imp_1d.imp)),4))
	l1 = pl.legend([p_pl,p_lg], ['$f(\phi) = Y\phi^{\delta}$', '$g(\phi)= a \log_{10}(1+b\phi)$'], loc=2, prop={'size':15})
	l2 = pl.legend([p_pl,p_lg], [leg_1 ,leg_2 ], loc=4, prop={'size':15})
	pl.gca().add_artist(l1)
	pl.subplots_adjust(bottom=0.15)
	pl.subplots_adjust(left=0.17)
	pl.savefig("../plot/" + name_plot + ".pdf")


	########################


def plot_tra_imp( name_plot ):

	# distance between axes and ticks
	pl.rcParams['xtick.major.pad']='8'
	pl.rcParams['ytick.major.pad']='8'

	# set latex font
	pl.rc('text', usetex=True)
	pl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})


	N_max_imp = 200

	pl.close('all')
	fig = pl.figure()
	ax = fig.add_subplot(1, 1, 1)
	pl.plot(range(N_max_imp),tmp_imp[:N_max_imp]/tmp_count[:N_max_imp], 'o', ms=3, color='SteelBlue')
	ax.set_xlabel('$t$')
	ax.set_ylabel('$\mathcal{I}(t)$')
	pl.grid()
	pl.subplots_adjust(bottom=0.15)
	pl.subplots_adjust(left=0.17)
	pl.savefig("../plot/" + name_plot + ".pdf")




###################################################################################################################
###################################################################################################################


										###################################
										####        Main code          ####
										###################################


###########################
## Reading data
##
## Calculating daily rate, participation rate, duration, impact and duration in minutes
###########################


print('Reading data...')
df_in = pd.read_csv('../data_frame/df_07_sel.csv',delimiter=';')
df_in['pi'] =  1. * df_in.Q/df_in.VD
df_in['eta'] = 1. * df_in.Q/df_in.VP
df_in['dur'] = 1. * df_in.VP/df_in.VD
df_in['imp'] = 1. * df_in.side * pl.log10(df_in.p_e/df_in.p_s) / df_in.sigma
df_in['dur_m'] = map(lambda x,y: x-y, map(extract_min,df_in.t_e), map(extract_min,df_in.t_s))	
	

###########################
## Generating a snapshot of the dataframe
##
## Selecting the metaorders executed on GE
## Extracting the trading date, the start and end minute of the execution
## Plotting 
###########################


print('Generating snapshot of the dataframe...') 
df_ge = df_in[df_in.symbol=='GE']
df_ge['day_trade'] = df_ge.t_s.apply(extract_day)
all_days = sorted(list(set(df_ge.day_trade))) 		# creating a list without repetition of the trading days 
df_ge['day_trade_n'] = find_pos(df_ge.day_trade, all_days)
df_ge['mm_s'] = df_ge.t_s.apply(extract_min)
df_ge['mm_e'] = df_ge.t_e.apply(extract_min)

if plotting_tmp:
	plot_ge('ge')


###########################
## Generating histogram of the daily rate Q/V_D
##
## Generating evenly populated bins by means of pl.percentile 
## Producing an histogram 
## Plotting
###########################


print('Generating histogram of the daily rate Q/V_D...')
n_bins_h_pi = 500
bin_end_h_pi = pl.percentile(df_in.pi,list(100.*pl.arange(n_bins_h_pi+1.)/(n_bins_h_pi)))
pdf_pi, bins_pi, patches = pl.hist(pl.array(df_in.pi), bin_end_h_pi, normed=1, histtype='step')
bins_pi_cent = (bins_pi[:-1] - bins_pi[1:])/2. + bins_pi[1:]	# finding the center of the bins

if plotting_tmp:
	plot_hist('stat_pi')


###########################
## Measuring temporary impact as a function of the daily rate Q/V_D
##
## Generating evenly populated bins of \pi by means of percentile
## Assigning to each metaorder the corresponding bin in df_in.fac_pi
## Evaluating the average daily rate and impact for each bin, standard deviation and counting by means of a groupby
## Fitting a power-law and a logarithmic function
## Plotting
###########################


print('Measuring temporary impact as a function of the daily rate Q/V_D ...')
n_bins_imp_pi = 30
bin_end_imp_pi = pl.percentile(df_in.pi,list(100.*pl.arange(n_bins_imp_pi+1.)/(n_bins_imp_pi)))
bin_end_imp_pi[-1] = bin_end_imp_pi[-1] + 0.00001	# fixing the last extreme of the bins
df_in['fac_pi'] = pl.digitize(df_in.pi,bin_end_imp_pi)
df_gp = df_in[['pi','imp','fac_pi']].groupby('fac_pi')
df_imp_1d = pd.concat([df_gp.mean(),df_gp.imp.std(),df_gp.imp.count()], axis=1)
df_imp_1d.columns = ['pi','imp','stdd','nn']

## Fitting temporary impact as a function of the daily rate Q/V_D
print('Fitting temporary impact as a function of the daily rate Q/V_D...')
# fitting a power-law function
ar_pl = [0., 0.3] 	# extremes of the grid of the starting points for the non-linear optimisation algorithm
br_pl = [0., 1.]	# extremes of the grid of the starting points for the non-linear optimisation algorithm
def ff_pl(x, a, b): return a * pow(x,b)
par_pl,vv_pl,chi_pl = fit_nonlin_1d_2p(ff_pl,df_imp_1d,ar_pl,br_pl)

# fitting a logarithmic function
ar_lg = [0., 0.1]	# extremes of the grid of the starting points for the non-linear optimisation algorithm
br_lg = [50., 500.]	# extremes of the grid of the starting points for the non-linear optimisation algorithm
def ff_lg(x, a, b): return a * pl.log10( 1.+b*x )
par_lg,vv_lg,chi_lg = fit_nonlin_1d_2p(ff_lg,df_imp_1d,ar_lg,br_lg)

if plotting_tmp:
	plot_tmp_imp('impact_temporary')


###########################
## Measuring transient impact as a function of time
##
## Selecting metaroders with respect to their duration and participation rate
## Reading for each metaorder the time series of the price during the execution 
## Calculating the price impact trajectory
## Averaging all the price impact trajectories
## Plotting
###########################


print('Measuring transient impact as a function of time...')
dur_min = 10
dur_max = 400
eta_min = 0.01
eta_max = 0.3

eq_list = df_in.symbol.unique()	# list of the equities present in the dataset

tmp_imp = pl.zeros(dur_max)		# empty vector for summing all the price impact trajectories
tmp_count = pl.zeros(dur_max)	# empty vector for summing the number of occurrencies in each minute 
for i_eq in eq_list: 
	print i_eq
	df_eq = df_in[df_in.symbol==i_eq]
	df_eq_sel = df_eq[(df_eq.dur_m>dur_min) & (df_eq.dur_m<dur_max) & (df_eq.eta>eta_min) & (df_eq.eta<eta_max)]
	df_eq_sel = df_eq_sel.set_index([range(df_eq_sel.shape[0])])

	ts_eq = pd.read_csv('../time_series/pr_07_'+i_eq+'.csv',  parse_dates = True, index_col = 0, date_parser=parse)

	for i in range(df_eq_sel.shape[0]):
		pr_during = pl.array( (ts_eq.V2[df_eq_sel.t_s[i]:df_eq_sel.t_e[i]]) )
		imp_during = df_eq_sel.side[i] * pl.log((pr_during / pr_during[0])) / df_eq_sel.sigma[i]
		# filling with zeros the price impact trajectory after the end of the execution
		imp_during_filled = pl.concatenate([imp_during,pl.zeros(dur_max-len(imp_during))])
		imp_counter = pl.concatenate([pl.ones(len(imp_during)),pl.zeros(dur_max-len(imp_during))])

		tmp_imp = tmp_imp + imp_during_filled
		tmp_count = tmp_count + imp_counter

if plotting_tmp:
	plot_tra_imp('impact_transient')



	




