import pylab as py
import scipy.optimize as optimize
import pandas as pd

execfile('functions.py')

## Reading data
year = '07'
path_in = '../data/data_'
df_in = pd.read_csv(path_in+year+'.dat',delimiter='|')

df_in['pi'] =  1. * df_in.Q/df_in.VD
df_in['imp'] = 1. * df_in.side * py.log10(df_in.p_e/df_in.p_s) / df_in.sigma

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
