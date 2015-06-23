import pylab as py
import scipy.optimize as optimize
import pandas as pd
from datetime import datetime

execfile('functions.py')
execfile('plotting.py')

plotting = False # set this variable to True if you want to produce the plots


###########################
## Reading data
###########################

print('Reading data...')
df_in = pd.read_csv('../data_frame/df_07_sel.csv',delimiter=';')

## Calculating daily rate, participation rate, duration, impact and duration in minutes
df_in['pi'] =  1. * df_in.Q/df_in.VD
df_in['eta'] = 1. * df_in.Q/df_in.VP
df_in['dur'] = 1. * df_in.VP/df_in.VD
df_in['imp'] = 1. * df_in.side * py.log10(df_in.p_e/df_in.p_s) / df_in.sigma
df_eq['dur_m'] = map(lambda x,y: x-y, map(extract_min,df_eq.t_e), map(extract_min,df_eq.t_s))	
	

###########################
## Generating a snapshot of the dataframe
###########################

print('Generating snapshot of the dataframe...') 
df_ge = df_in[df_in.symbol=='GE'] 							# selecting metaorders executed on GE
df_ge['day_trade'] = df_ge.t_s.apply(extract_day) 			# extracting the trading date
all_days = sorted(list(set(df_ge.day_trade))) 				# creating a list without repetition of the trading days 
df_ge['day_trade_n'] = find_pos(df_ge.day_trade, all_days)	# associating an increasing number to the trading date, for plotting purposes
df_ge['mm_s'] = df_ge.t_s.apply(extract_min)				# extracting the metaorder starting minute
df_ge['mm_e'] = df_ge.t_e.apply(extract_min)				# extracting the metaorder ending minute

if plotting:
	plot_ge() 												# plotting function


###########################
## Generating histogram of the daily rate Q/V_D
###########################

print('Generating histogram of the daily rate Q/V_D...')
n_bins_h_pi = 500 																					# setting the number of bins of the histogram
bin_end_h_pi = py.percentile(df_in.pi,list(100.*py.arange(n_bins_h_pi+1.)/(n_bins_h_pi)))			# finding the extremes of evenly-populated bins by means of percentyle
pdf_pi, bins_pi, patches = py.hist(py.array(df_in.pi), bin_end_h_pi, normed=1, histtype='step')		# histogram
bins_pi_cent = (bins_pi[:-1] - bins_pi[1:])/2. + bins_pi[1:]										# finding the center of the bins

if plotting:
	plot_hist()																						# plotting function


###########################
## Measuring temporary impact as a funtion of the daily rate Q/V_D
###########################

print('Measuring temporary impact as a funtion of the daily rate Q/V_D ...')
n_bins_imp_pi = 30																					# setting the number of the grouping bins	
bin_end_imp_pi = py.percentile(df_in.pi,list(100.*py.arange(n_bins_imp_pi+1.)/(n_bins_imp_pi)))		# finding the extremes of evenly-populated bins by means of percentyle
bin_end_imp_pi[-1] = bin_end_imp_pi[-1] + 0.00001													# fixing the last extreme of the bins
df_in['fac_pi'] = py.digitize(df_in.pi,bin_end_imp_pi)												# assigning to each metaorder the corresponding bin
df_gp = df_in[['pi','imp','fac_pi']].groupby('fac_pi')												# groupby according to the assigned bins
df_imp_1d = pd.concat([df_gp.mean(),df_gp.imp.std(),df_gp.imp.count()], axis=1)						# evaluating the average daily rate and impact for each bin, standard deviation and counting 
df_imp_1d.columns = ['pi','imp','stdd','nn']

## Fitting temporary impact as a funtion of the daily rate Q/V_D
print('Fitting temporary impact as a funtion of the daily rate Q/V_D...')
# fitting a power-law function
ar_pl = [0., 0.3] 																					# setting the extremes of the grid of the starting points for the non-linear optimisation algorithm
br_pl = [0., 1.]																					# setting the extremes of the grid of the starting points for the non-linear optimisation algorithm
def ff_pl(x, a, b): return a * pow(x,b)
par_pl,vv_pl,chi_pl = fit_nonlin_1d_2p(ff_pl,df_imp_1d,ar_pl,br_pl)									# optimisation algorithm

# fitting a logarithmic function
ar_lg = [0., 0.1]																					# setting the extremes of the grid of the starting points for the non-linear optimisation algorithm
br_lg = [50., 500.]																					# setting the extremes of the grid of the starting points for the non-linear optimisation algorithm
def ff_lg(x, a, b): return a * py.log10( 1.+b*x )
par_lg,vv_lg,chi_lg = fit_nonlin_1d_2p(ff_lg,df_imp_1d,ar_lg,br_lg)									# optimisation algorithm

if plotting:
	plot_tmp_imp()


###########################
## Measuring transient impact as a function of time
###########################

print('Measuring transient impact as a function of time...')
dur_min = 10 						# setting the minimum duration of the metaorders in the analysis
dur_max = 400						# setting the maximum duration of the metaorders in the analysis
eta_min = 0.01 						# setting the minimum participation rate of the metaorders in the analysis 	
eta_max = 0.3 						# setting the maximum participation rate of the metaorders in the analysis

eq_list = df_in.symbol.unique() 	# generating the list of the equity in present in the dataset

tmp_imp = py.zeros(dur_max)			# creating an empty vecton for summing all the price impact trajectories
tmp_count = py.zeros(dur_max)		# creating an empty vecton for counting the number of occurrencies in each minute 
for i_eq in eq_list: 
	print i_eq																											# cicling on the equities in the dataset				
	df_eq = df_in[df_in.symbol==i_eq]   																				# selecting the metaorders of the selected equity
	df_eq_sel = df_eq[(df_eq.dur_m>dur_min) & (df_eq.dur_m<dur_max) & (df_eq.eta>eta_min) & (df_eq.eta<eta_max)] 		# selecting metaorders that respect the constraints of duration and participation rate
	df_eq_sel = df_eq_sel.set_index([range(df_eq_sel.shape[0])]) 														# resetting the index of the resulting dataset

	ts_eq = pd.read_csv('../time_series/pr_07_'+i_eq+'.csv',  parse_dates = True, index_col = 0, date_parser=parse) 	# reading the price time series of the selected equity

	for i in range(df_eq_sel.shape[0]): 																# ciclying on the metaorders in df_eq_sel
		pr_during = py.array( (ts_eq.V2[df_eq_sel.t_s[i]:df_eq_sel.t_e[i]]) ) 							# extracting the price time sereies of the selected metaorder (during)
		imp_during = df_eq_sel.side[i] * py.log((pr_during / pr_during[0])) / df_eq_sel.sigma[i]		# measuring the price impact trajectory
		imp_during_filled = py.concatenate([imp_during,py.zeros(dur_max-len(imp_during))])				# filling with zeros the price impact trajectory after the end of the execution
		imp_counter = py.concatenate([py.ones(len(imp_during)),py.zeros(dur_max-len(imp_during))])		# setting 1 to the counter during the execution and 0 after the execution

		tmp_imp = tmp_imp + imp_during_filled		# adding the price impact trajectory to the average
		tmp_count = tmp_count + imp_counter			# adding the counter 

if plotting:
	plot_tra_imp()



	




