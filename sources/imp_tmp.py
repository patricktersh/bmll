import pylab as py
import scipy.optimize as optimize
import pandas as pd

execfile('functions.py')

## Reading data
year = '07'
path_in = '../data/data_'
df_in = pd.read_csv(path_in+year+'_short.dat',delimiter='|')
df_in['pi'] =  1. * df_in.Q/df_in.VD
df_in['eta'] = 1. * df_in.Q/df_in.VP
df_in['dur'] = 1. * df_in.VP/df_in.VD
df_in['imp'] = 1. * df_in.side * py.log10(df_in.p_e/df_in.p_s) / df_in.sigma


###########################

aapl = df_in[df_in.symbol=='AAPL']
aapl['day_trade'] = aapl.t_s.apply(extract_day)
all_days = sorted(list(set(aapl.day_trade)))
aapl['day_trade_n'] = find_pos(aapl.day_trade, all_days)
aapl['mm_s'] = aapl.t_s.apply(extract_min)
aapl['mm_e'] = aapl.t_e.apply(extract_min)


###########################


n_bins_h_pi = 10
bin_end_h_pi = py.percentile(df_in.pi,list(100.*py.arange(n_bins_h_pi+1.)/(n_bins_h_pi)))
pdf_pi, bins_pi, patches = py.hist(py.array(df_in.pi), bin_end_h_pi, normed=1, histtype='step')
bins_pi_cent = (bins_pi[:-1] - bins_pi[1:])/2. + bins_pi[1:]


###########################


n_bins = 30
bin_end = py.percentile(df_in.pi,list(100.*py.arange(n_bins+1.)/(n_bins)))
bin_end[len(bin_end)-1] = bin_end[len(bin_end)-1] + 0.00001
df_in['fac'] = py.digitize(df_in.pi,bin_end)
df_gp = df_in[['pi','imp','fac']].groupby('fac')	
df_out = pd.concat([df_gp.mean(),df_gp.std().imp,df_gp.count().imp], axis=1)
df_out.columns = ['pi','imp','stdd','nn']

# fitting a power-law function
ar_pl = [0., 0.3]
br_pl = [0., 1.]
def ff_pl(x, a, b): return a * pow(x,b)
par_pl,vv_pl,chi_pl = fit_nonlin_1d_2p(ff_pl,df_out,ar_pl,br_pl)

# fitting a logarithmic function
ar_lg = [0., 0.1]
br_lg = [50., 500.]
def ff_lg(x, a, b): return a * py.log10( 1.+b*x )
par_lg,vv_lg,chi_lg = fit_nonlin_1d_2p(ff_lg,df_out,ar_lg,br_lg)


###########################







execfile('plotting.py')




