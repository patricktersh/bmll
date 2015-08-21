__author__ = 'eliazarinelli'

import functions as fn
import pandas as pd
import pylab as pl
from datetime import datetime

ANCERNO_COL_NAMES = ['symbol', 'side', 'Q', 'VP', 'VD', 'sigma', 't_s', 't_e', 'p_s', 'p_e', 'pi', 'eta', 'dur', 'imp']



class Filter():
    def __init__(self, \
                 symbols = ['AIG', 'BAC', 'C', 'CSCO', 'GE', 'JPM', 'MRK', 'MSFT', 'PG', 'XOM'], \
                 months = ['2007-01', '2007-02', '2007-03', '2007-04', '2007-05', '2007-06', \
                           '2007-07', '2007-08', '2007-09', '2007-10', '2007-11', '2007-12', ], \
                 t_s = ['08:00:00','18:00:00'], \
                 t_e = ['08:00:00','18:00:00'], \
                 pi = [0.,1.], \
                 eta = [0.,1.], \
                 dur = [0.,1.]):
        self.symbols_list = symbols
        self.months_list = months
        self.extremes = {'t_s': t_s, \
                         't_e': t_e, \
                         'pi' : pi, \
                         'eta': eta, \
                         'dur': dur}
    def __str__(self):
        return 'Filter \nsymbols : %s \nmonths  : %s \nt_s: %s \nt_e: %s \npi : %s \neta: %s \ndur: %s \n' % \
               (self.symbols_list, self.months_list, self.extremes['t_s'], self.extremes['t_e'], \
                self.extremes['pi'], self.extremes['eta'], self.extremes['dur'])

    #### Methods for setting each single field ####

    def set_symbols(self, list_in):
        self.symbols_list = list_in

    def set_months(self, list_in):
        self.months_list = list_in

    def set_t_s_min(self, v_in):
        self.extremes['t_s'][0] = v_in

    def set_t_s_max(self, v_in):
        self.extremes['t_s'][1] = v_in

    def set_t_e_min(self, v_in):
        self.extremes['t_e'][0] = v_in

    def set_t_e_max(self, v_in):
        self.extremes['t_e'][1] = v_in

    def set_pi_min(self, v_in):
        self.extremes['pi'][0] = v_in

    def set_pi_max(self, v_in):
        self.extremes['pi'][1] = v_in

    def set_eta_min(self, v_in):
        self.extremes['eta'][0] = v_in

    def set_eta_max(self, v_in):
        self.extremes['eta'][1] = v_in

    def set_dur_min(self, v_in):
        self.extremes['dur'][0] = v_in

    def set_dur_max(self, v_in):
        self.extremes['dur'][1] = v_in


class BinnedData2D:

    def __init__(self, n_bins = 10):
        self.n_bins = n_bins
        self.source = ""
        self.filter = Filter()
        self.data = pd.DataFrame(data = pl.ones((self.n_bins,4)), columns = ['pi','imp','stdd','nn'])

    def get_n_bins(self):
        return self.n_bins

    def get_source(self):
        return self.source

    def get_filter(self):
        return self.filter

    def get_data(self):
        return self.data


class AncernoDatabase():

    def __init__(self, filter = None):

        self.source = ""
        # building an empty pandas dataframe
        self.db_raw = pd.DataFrame(columns = ANCERNO_COL_NAMES)
        self.db_fil = self.db_raw
        self.filter = filter or Filter()

    def import_dataset_csv(self, name_csv = '../data_frame/test_dataframe.csv'):
        self.source = name_csv
        # importing the data from a csv file
        self.db_raw = pd.read_csv(name_csv, delimiter=';')
        # calculating derivate quantities: pi, eta, dur, imp
        self.db_raw['pi'] =  1. * self.db_raw.Q/self.db_raw.VD
        self.db_raw['eta'] = 1. * self.db_raw.Q/self.db_raw.VP
        self.db_raw['dur'] = 1. * self.db_raw.VP/self.db_raw.VD
        #self.db['imp'] = 1. * self.db.side * pl.log10(self.db.p_e/self.db.p_s) / self.db.sigma
        self.db_raw['imp'] = 1. * self.db_raw.side * pl.log(self.db_raw.p_e/self.db_raw.p_s) / self.db_raw.sigma

    #### Methods for getting each single field ####

    def get_symbol(self):
        return self.db_fil.symbol

    def get_side(self):
        return self.db_fil.side

    def get_Q(self):
        return self.db_fil.Q

    def get_VP(self):
        return self.db_fil.VP

    def get_VD(self):
        return self.db_fil.VD

    def get_sigma(self):
        return self.db_fil.sigma

    def get_t_s(self):
        return self.db_fil.t_s

    def get_t_e(self):
        return self.db_fil.t_e

    def get_p_s(self):
        return self.db_fil.p_s

    def get_p_e(self):
        return self.db_fil.p_e

    def get_pi(self):
        return self.db_fil.pi

    def get_eta(self):
        return self.db_fil.eta

    def get_dur(self):
        return self.db_fil.dur

    def get_imp(self):
        return self.db_fil.imp

    #### Method for setting and apply the filter ####

    def get_filter(self):
        return self.filter

    def apply_filter(self, filter = None):

        self.filter = filter or Filter()

        #### Applying filter ####

        # filter_symbol: the traded stock must be in the filter list
        filter_symbol = fn.is_in(self.db_raw['symbol'], self.filter.symbols_list)

        # filter_month: the day of the execution must be in the filter list, the format of the list is YYYY-MM
        tmp_0 = self.db_raw['t_s']
        tmp_1 = tmp_0.apply(fn.extract_ym)
        filter_month = fn.is_in(tmp_1, self.filter.months_list)

        # filter_t_s: the starting time of the execution must be within the filter extremes
        tmp_0 = self.db_raw['t_s']
        tmp_1 = tmp_0.apply(fn.extract_min)
        tmp_2 = map(fn.extract_min_short,self.filter.extremes['t_s'])
        filter_t_s_0 = tmp_1 > tmp_2[0]
        filter_t_s_1 = tmp_1 < tmp_2[1]

        # filter_t_e: the ending time of the execution must be within the filter extremes
        tmp_0 = self.db_raw['t_e']
        tmp_1 = tmp_0.apply(fn.extract_min)
        filter_t_e_0 = tmp_1 > fn.extract_min_short(self.filter.extremes['t_e'][0])
        filter_t_e_1 = tmp_1 < fn.extract_min_short(self.filter.extremes['t_e'][1])

        # filter_pi: the daily fraction must be within the filter extremes
        filter_pi_0 = self.db_raw['pi'] > self.filter.extremes['pi'][0]
        filter_pi_1 = self.db_raw['pi'] < self.filter.extremes['pi'][1]

        # filter_eta: the participation rate must be within the filter extremes
        filter_eta_0 = self.db_raw['eta'] > self.filter.extremes['eta'][0]
        filter_eta_1 = self.db_raw['eta'] < self.filter.extremes['eta'][1]

        # filter_dur: the duration must be within the filter extremes
        filter_dur_0 = self.db_raw['dur'] > self.filter.extremes['dur'][0]
        filter_dur_1 = self.db_raw['dur'] < self.filter.extremes['dur'][1]

        # applying all filters
        filter_all = filter_t_s_0 & filter_t_s_1 & filter_t_e_0 & filter_t_e_1 & filter_pi_0 & filter_pi_1 \
                     & filter_eta_0 & filter_eta_1 & filter_dur_0 & filter_dur_1 & filter_symbol & filter_month
        self.db_fil = self.db_raw.loc[filter_all,:]

    def get_binned_data_2d(self, n_bins = 10):

        bd_in = BinnedData2D()
        bd_in.source = self.source
        bd_in.filter = self.filter
        bd_in.n_bins = n_bins

        #### Generating binned data ####

        # Extracting pi and imp
        database_reduced = self.db_fil.loc[:,['pi','imp']]

        # Generating the bin extremes
        bin_end_imp_pi = pl.percentile(database_reduced.pi,list(100.*pl.arange(bd_in.n_bins+1.)/(bd_in.n_bins)))

        # Adjusting the last bin extreme
        bin_end_imp_pi[-1] = bin_end_imp_pi[-1] + 0.00001

        # Assigning each point to a bin
        database_reduced['fac_pi'] = pl.digitize(database_reduced.pi,bin_end_imp_pi)

        # Using a groupby in order to generate average pi and imp for each bin, assigning the output to df_imp
        df_gp = database_reduced[['pi','imp','fac_pi']].groupby('fac_pi')
        #df_imp = df_gp.mean()
        df_imp = pd.concat([df_gp.mean(),df_gp.imp.std(),df_gp.imp.count()], axis=1)
        df_imp.columns = ['pi','imp','stdd','nn']

        # Setting the data
        bd_in.data = df_imp

        return bd_in

if False:
    # generating a filter that does not filter
    fil = Filter()

    # creating an empty database
    db = AncernoDatabase()

    #importing the test dataset from csv
    db.import_dataset_csv('../data_frame/test_impact.csv')

    # applying the fitler
    db.apply_filter(fil)

    # getting 2d binned data
    data = db.get_binned_data_2d(10)

    print(data.get_source())
    print(data.get_n_bins())
    print(data.get_filter())
    print(data.get_data())








class Impact2D:

    def __init__(self):
        self.n_bins = 10
        self.data = pd.DataFrame(data = 0.1*pl.ones((self.n_bins,4)), columns = ['pi','imp','stdd','nn'])
        self.functions = {'power law': fn.ff_pl, 'logarithm': fn.ff_lg}
        self.parameters = {'power law': [1., 1.], 'logarithm': [1., 1.]}
        self.errors = {'power law': [1., 1.], 'logarithm': [1., 1.]}
        self.chi = {'power law': 1., 'logarithm': 1.}
        self.filter = Filter()

    def __str__(self):
       return 'Power Law \nImp(Q/V) = Y*(Q/V)^delta\nY = %f, delta = %f \nchi = %f\n\nLogarithm \nImp(Q/V) = a*log[1+b*(Q/V)]\na = %f, b = %f\nchi = %f'  \
              % (self.parameters['power law'][0], self.parameters['power law'][1],self.chi['power law'],self.parameters['logarithm'][0], self.parameters['logarithm'][1],self.chi['logarithm'])

    def get_data(self):
        return self.data

    def get_fun(self, key_in):
        return self.functions[key_in]

    def get_par(self, key_in):
        return self.parameters[key_in]

    def get_err(self, key_in):
        return self.errors[key_in]

    def get_chi(self, key_in):
        return self.chi[key_in]

    def set_filter(self, filter):
        self.filter = filter

    def calibrate_impact(self, database, n_bins):

        self.n_bins = n_bins

        #### Applying filter ####

        # filter_symbol: the traded stock must be in the filter list
        filter_symbol = fn.is_in(database.get_symbol(), self.filter.symbols_list)

        # filter_month: the day of the execution must be in the filter list, the format of the list is YYYY-MM
        tmp_0 = database.get_t_s()
        tmp_1 = tmp_0.apply(fn.extract_ym)
        filter_month = fn.is_in(tmp_1, self.filter.months_list)

        # filter_t_s: the starting time of the execution must be within the filter extremes
        tmp_0 = database.get_t_s()
        tmp_1 = tmp_0.apply(fn.extract_min)
        tmp_2 = map(fn.extract_min_short,self.filter.extremes['t_s'])
        filter_t_s_0 = tmp_1 > tmp_2[0]
        filter_t_s_1 = tmp_1 < tmp_2[1]

        # filter_t_e: the ending time of the execution must be within the filter extremes
        tmp_0 = database.get_t_e()
        tmp_1 = tmp_0.apply(fn.extract_min)
        filter_t_e_0 = tmp_1 > fn.extract_min_short(self.filter.extremes['t_e'][0])
        filter_t_e_1 = tmp_1 < fn.extract_min_short(self.filter.extremes['t_e'][1])

        # filter_pi: the daily fraction must be within the filter extremes
        filter_pi_0 = database.get_pi() > self.filter.extremes['pi'][0]
        filter_pi_1 = database.get_pi() < self.filter.extremes['pi'][1]

        # filter_eta: the participation rate must be within the filter extremes
        filter_eta_0 = database.get_eta() > self.filter.extremes['eta'][0]
        filter_eta_1 = database.get_eta() < self.filter.extremes['eta'][1]

        # filter_dur: the duration must be within the filter extremes
        filter_dur_0 = database.get_dur() > self.filter.extremes['dur'][0]
        filter_dur_1 = database.get_dur() < self.filter.extremes['dur'][1]

        # applying all filters
        filter_all = filter_t_s_0 & filter_t_s_1 & filter_t_e_0 & filter_t_e_1 & filter_pi_0 & filter_pi_1 \
                     & filter_eta_0 & filter_eta_1 & filter_dur_0 & filter_dur_1 & filter_symbol & filter_month
        database_filtered = database.db.loc[filter_all,:]

        #### Generating binned data ####

        # Extracting pi and imp
        database_reduced = database_filtered.loc[:,['pi','imp']]

        # Generating the bin extremes
        bin_end_imp_pi = pl.percentile(database_reduced.pi,list(100.*pl.arange(self.n_bins+1.)/(self.n_bins)))

        # Adjusting the last bin extreme
        bin_end_imp_pi[-1] = bin_end_imp_pi[-1] + 0.00001

        # Assigning each point to a bin
        database_reduced['fac_pi'] = pl.digitize(database_reduced.pi,bin_end_imp_pi)

        # Using a groupby in order to generate average pi and imp for each bin, assigning the output to df_imp
        df_gp = database_reduced[['pi','imp','fac_pi']].groupby('fac_pi')
        #df_imp = df_gp.mean()
        df_imp = pd.concat([df_gp.mean(),df_gp.imp.std(),df_gp.imp.count()], axis=1)
        df_imp.columns = ['pi','imp','stdd','nn']

        # Setting the data
        self.data = df_imp

        #### Estimating parameters ####

        ar = [0., 0.3] 	# extremes of the grid of the starting points for the non-linear optimisation algorithm
        br = [0., 1.]	# extremes of the grid of the starting points for the non-linear optimisation algorithm
        parameters,errors,chi = fn.fit_nonlin_1d_2p(self.functions['power law'],self.data,ar,br)
        self.parameters['power law'] = parameters
        self.errors['power law'] = [errors[0][0],errors[1][1]]
        self.chi['power law'] = chi

        ar = [0., 0.3] 	# extremes of the grid of the starting points for the non-linear optimisation algorithm
        br = [0., 1.]	# extremes of the grid of the starting points for the non-linear optimisation algorithm
        parameters,errors,chi = fn.fit_nonlin_1d_2p(self.functions['logarithm'],self.data,ar,br)
        self.parameters['logarithm'] = parameters
        self.errors['logarithm'] = [errors[0][0],errors[1][1]]
        self.chi['logarithm'] = chi

    def plot_impact(self):

        # generating points for functions plotting
        x_plf = pow(10,pl.linspace(-6,0,1000))
        y_plf_pl = self.functions['power law'](x_plf, self.get_par('power law')[0], self.get_par('power law')[1])
        y_plf_lg = self.functions['logarithm'](x_plf, self.get_par('logarithm')[0], self.get_par('logarithm')[1])

        # set latex font
        pl.rc('text', usetex=True)
        pl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})
        pl.rcParams['xtick.major.pad']='8'
        pl.rcParams['ytick.major.pad']='8'

        # plotting
        pl.clf()
        p_pl, = pl.plot(x_plf, y_plf_pl, ls='--', color='Red')
        p_lg, = pl.plot(x_plf, y_plf_lg, ls='-', color='RoyalBlue')
        p_points, = pl.plot(self.get_data().pi, self.get_data().imp,'.', color='Black',ms=10)

        pl.xscale('log')
        pl.yscale('log')
        pl.xlabel('$\phi$')
        pl.ylabel('$\mathcal{I}_{tmp}(\Omega=\{ \phi \})$')
        pl.grid()
        pl.axis([0.00001,1,0.0001,0.1])

        l_00 = '$\hat{Y} = $' + str("%.4f" % round(self.get_par('power law')[0],4)) + '$\pm$' + str("%.4f" % round(self.get_err('power law')[0],4))
        l_01 = '$\hat{\delta} = $' + str("%.4f" % round(self.get_par('power law')[1],4)) + '$\pm$' + str("%.4f" % round(self.get_err('power law')[1],4))
        l_02 = '$E_{RMS} = $' + str("%.4f" % round(pl.sqrt(self.get_chi('power law')/self.n_bins),4))
        leg_0 = l_00 + " " + l_01 + " " + l_02

        l_10 = '$\hat{a} = $' + str("%.4f" % round(self.get_par('logarithm')[0],4)) + '$\pm$' + str("%.4f" % round(self.get_err('logarithm')[0],4))
        l_11 = '$\hat{b} = $' + str("%.4f" % round(self.get_par('logarithm')[1],4)) + '$\pm$' + str("%.4f" % round(self.get_err('logarithm')[1],4))
        l_12 = '$E_{RMS} = $' + str("%.4f" % round(pl.sqrt(self.get_chi('logarithm')/self.n_bins),4))
        leg_1 = l_10 + " " + l_11 + " " + l_12
        l1 = pl.legend([p_pl,p_lg], ['$f(\phi) = Y\phi^{\delta}$', '$g(\phi)= a \log_{10}(1+b\phi)$'], loc=2, prop={'size':15})
        l2 = pl.legend([p_pl,p_lg], [leg_0 ,leg_1 ], loc=4, prop={'size':15})
        pl.gca().add_artist(l1)
        pl.subplots_adjust(bottom=0.15)
        pl.subplots_adjust(left=0.17)
        pl.show()


class Order:

    def __init__(self, symbol = None, day = None, quantity = None):
        self.symbol = symbol or 'AAPL'
        self.day = day or '2010-02-01'
        self.quantity = quantity or 1000.
        self.volatility_est = 0.01
        self.volume_est = 100000.
        self.impact_exp = 0.
        self.impact_model = 'power law'

    def __str__(self):

        sep = '\n----------\n'
        p_0 = 'ORDER:'
        p_1 = 'Symbol: %s\nDay: %s\nQuantity: %s' \
               % (self.symbol, self.day, self.quantity)
        p_2 = 'Estimated Volume: %f\nEstimated Volatility: %f\nExpected Daily Rate: %f' % (self.volume_est,self.volatility_est, self.quantity/self.volume_est)
        p_3 = 'Impact Model: %s\nPredicted Impact (bp): %f' % (self.impact_model,self.impact_exp)

        return p_0 + sep + p_1 + sep + p_2 + sep + p_3 + sep

    def estimate_vol_vol(self):

        # dummy time series: must be replaced by a reader from an external source
        rng = pd.date_range(start = '2010-01-01', end = '2011-01-01', freq='D')
        tmp_0 = 100000. + 10000.*pl.randn(len(rng))
        ts_volume = pd.Series(tmp_0, index=rng)
        tmp_1 = 0.02 + 0.002*pl.randn(len(rng))
        ts_volatility = pd.Series(tmp_1, index=rng)

        # estimation of the daily volume and of the daily volatility as a flat average of the previuos n_days_mav
        n_days_mav = 10
        period_start = pd.to_datetime('2010-03-01') + pd.DateOffset(days=-(n_days_mav+1))
        period_end = pd.to_datetime('2010-03-01') + pd.DateOffset(days=-1)
        self.volume_est = ts_volume[period_start:period_end].mean()
        self.volatility_est = ts_volatility[period_start:period_end].mean()


    def calculate_impact_2d(self, impact_2d = None, impact_model = None):
        impact_2d = impact_2d or Impact2D()
        self.impact_model = impact_model or 'power law'
        par = impact_2d.get_par(impact_model)
        fn = impact_2d.get_fun(impact_model)
        self.impact_exp = (pl.exp(self.volatility_est * fn(self.quantity/self.volume_est,par[0],par[1]))-1.)*10000.


if False:
    a = Order(symbol='AAPL', day='2010-03-01', quantity=10000)
    a.estimate_vol_vol()
    a.calculate_impact_2d(impact_model = 'power law')
    print(a)







#class Database2D():
#    def __init__(self, db_info=None):
#        self.db = pd.DataFrame(data = pl.ones((100,2)), columns = ['pi','imp'])
#        self.db_info = db_info or DatabaseInfo()



#a = Impact2D()
#print(a)

if False:
    db = AncernoDatabase()
    db.import_dataset_csv('../data_frame/test_impact.csv')

    fil = Filter()

    imp = Impact2D()
    imp. set_filter(fil)
    imp.calibrate_impact(db, 20)
    #imp.plot_impact()

    a = Order(symbol='AAPL', day='2010-03-01', quantity=10000)
    a.estimate_vol_vol()
    a.calculate_impact_2d(impact_model = 'power law')
    print(a)

    #print(imp)



