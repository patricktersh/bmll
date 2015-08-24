__author__ = 'eliazarinelli'

import functions as fn
import pandas as pd
import pylab as pl
import scipy.optimize as optimize
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

    def __init__(self):
        # attribute set equal to the number of bins used in the binning procedure
        self.n_bins = 10
        # attribute set equal to the csv file used to fill the dataset from which the binned data are generated
        self.source = ""
        # attribute set equal to the filter used to filter the database
        self.filter = Filter()
        # attribute containing the binned data 2D
        self.data = pd.DataFrame(data = pl.ones((self.n_bins,4)), columns = ['pi','imp','stdd','nn'])

    #### Methods for getting each single attribute ####

    def get_n_bins(self):
        return self.n_bins

    def get_source(self):
        return self.source

    def get_filter(self):
        return self.filter

    def get_data(self):
        return self.data


class AncernoDatabase():

    def __init__(self):

        # attribute describing the csv file that has been used in the inport_database_csv
        self.source = ""
        # db_raw is the database imported from the csv file, it is not modified by the application of the filter
        self.db_raw = pd.DataFrame(columns = ANCERNO_COL_NAMES)
        # db_fil is the database obtained from db_raw after applying a filter
        self.db_fil = self.db_raw
        # filter is used to obtain db_fil from db_raw via the method apply_filter
        self.filter = Filter()

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

    def get_filter(self):
        return self.filter

    #### Method for importing the dataset raw from a csv file ####

    def import_dataset_csv(self, name_csv = '../data_frame/test_dataframe.csv'):

        # setting the source equal to the import csv file name
        self.source = name_csv
        # importing db_raw from the csv file
        self.db_raw = pd.read_csv(name_csv, delimiter=';')
        # calculating derivate quantities: pi, eta, dur, imp
        self.db_raw['pi'] =  1. * self.db_raw.Q/self.db_raw.VD
        self.db_raw['eta'] = 1. * self.db_raw.Q/self.db_raw.VP
        self.db_raw['dur'] = 1. * self.db_raw.VP/self.db_raw.VD
        self.db_raw['imp'] = 1. * self.db_raw.side * pl.log(self.db_raw.p_e/self.db_raw.p_s) / self.db_raw.sigma


    #### Method for applying the filter ####

    def apply_filter(self, filter = None):

        # setting the attribute filter equal to the filter that the method receive as input
        self.filter = filter or Filter()


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

        # setting db_fil equal to the bd_raw after filtering the rows via filter_all
        self.db_fil = self.db_raw.loc[filter_all,:]


    #### Generating binned data ####

    def get_binned_data_2d(self, n_bins = 10):

        # generating an instance of the class BinnedData2D
        bd_in = BinnedData2D()

        # setting the attribute source of the output equal to the source of the dataset
        bd_in.source = self.source

        # setting the filter of the output equat to the filter of the dataset
        bd_in.filter = self.filter

        # setting the number of bins of the output equal to the number of bin that the method receives as an imput
        bd_in.n_bins = n_bins

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
        df_imp = pd.concat([df_gp.mean(),df_gp.imp.std(),df_gp.imp.count()], axis=1)
        df_imp.columns = ['pi','imp','stdd','nn']

        # Setting the data of the output equal to the result of the binning procedure
        bd_in.data = df_imp

        # returning the filled instance of the class BinnedData2D
        return bd_in




class ImpactModel2D:

    def __init__(self, n_parameters):
        # the number of parameters of the model should be set when the constructor is invoked
        self.n_parameters = n_parameters
        # model is a function that is set by the user
        # the number of function input must be n_parameters +1
        self.model = None
        # model parameters that must be calibrated
        self.parameters = [1.]*self.n_parameters
        # error on the parameters from the calibration procedure
        self.errors = [1.]*self.n_parameters
        # goodness of fit
        self.chi = float("inf")
        # list containing the extreme starting points of each parameter for the minimisation algorithm
        self.extremes = [[0.,1.]]*self.n_parameters
        # number of step of the between the extremes
        self.n_step = 5
        # the binned data that will be used as input to the fitting procedure
        self.db = BinnedData2D()
        # the description of the model that will appear in the plot
        self.latex = ""

    def set_model(self, model):
        self.model = model

    def set_extremes(self, extremes):
        if len(extremes) == self.n_parameters:
            self.extremes = extremes
        else:
            print "Wrong input: the input length does not match the number of model parameters"

    def set_latex(self,string_in):
        self.latex = string_in

    def get_parameters(self):
        return self.parameters

    def get_errors(self):
        return self.errors

    def get_chi(self):
        return self.chi

    def calibrate_model(self, data):
        self.db = data
        n_step = self.n_step
        n_par = self.n_parameters

        # Due to the fact that the fitting function could be non-linear function of the fitting parameters
        # the minimisation procedure could be stuck in a local minimum.
        # In order to avoid this issue we introduce a minimisation procedure from several fitting-parameters
        # starting points and we keep as best-fitting-parameters the one with smallest goodness of fit (chi).

        # Each column of mat_start_point contains the starting parameters
        # for the minimisation algorithm of each parameter: it is an_step x n_par matrix
        mat_start_point  = pl.ones((n_step, n_par))
        for i in range(n_par):
            mat_start_point[:,i] = pl.linspace(self.extremes[i][0],self.extremes[i][1],n_step)

        # Within the for loop we move along all the starting points of the grid
        for i in range(n_step**n_par):
            # To each integer of the for loop we associate the n_par-dimensional point of the grid-starting point
            starting_point = fn.generate_starting_point(i,mat_start_point,n_par,n_step)
            # For each grid-strting point we associate
            out_fit = optimize.curve_fit(self.model, data.get_data().pi, data.get_data().imp, starting_point,\
                                         pl.array(data.get_data().stdd)/pl.array(pl.sqrt(data.get_data().nn)))
            # defining the local best-fitting parameters
            local_par = out_fit[0]
            # defining the local errors associated to the best-fitting parameters
            local_err = []
            for i in range(n_par):
                local_err = local_err + [out_fit[1][i,i]]
            # calculating the goodness of the fit
            model_prediction = self.model(data.get_data().pi, *local_par)
            local_chi = sum(pow(data.get_data().imp - model_prediction,2))
            # if the local goodness of fit improves the ones reached before, I keep the new one
            if local_chi < self.chi:
                self.chi = local_chi
                self.parameters = local_par
                self.errors = local_err

    def plot_model(self):

        param_names = ['a','b','c','d','e']


        # settings
        pl.clf()
        pl.rc('text', usetex=True)
        pl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})
        pl.rcParams['xtick.major.pad']='8'
        pl.rcParams['ytick.major.pad']='8'
        pl.xscale('log')
        pl.yscale('log')
        pl.xlabel('$\phi = Q/V_D$')
        pl.ylabel('$\mathcal{I}_{tmp}( \phi)$')
        pl.grid()
        pl.axis([0.00001,1,0.0001,0.1])
        pl.subplots_adjust(bottom=0.15)
        pl.subplots_adjust(left=0.17)

        # generating points for functions plotting
        x_model = pow(10,pl.linspace(-6,0,1000))
        y_model = self.model(x_model, *self.parameters)
        p_model, = pl.plot(x_model, y_model, ls='--', color='Red')
        p_model.set_label(self.latex)
        l_1 = pl.legend(loc=2, prop={'size':15})
        ll = ''
        for i in range(self.n_parameters):
            ll = ll + param_names[i] + ' = ' + str("%.4f" % round(self.parameters[i],4)) + '$\pm$' + str("%.4f" % round(self.errors[i],4)) + ' '
        l_2 = pl.legend([ll], loc=4, prop={'size':15})
        pl.gca().add_artist(l_1)

        pl.plot(self.db.get_data().pi, self.db.get_data().imp,'.', color='Black',ms=10)
        pl.show()




if True:
    # generating a filter that does not filter
    fil = Filter()

    # creating an empty database
    db = AncernoDatabase()

    #importing the test dataset from csv
    db.import_dataset_csv('../data_frame/test_impact.csv')

    # applying the fitler
    db.apply_filter(fil)

    # getting 2d binned data
    data = db.get_binned_data_2d(50)

    # generating a model
    mod = ImpactModel2D(n_parameters=2)
    mod.set_model(fn.ff_pl)
    mod.set_latex('$f(\phi;a,b) = a \cdot \phi^{b}$')
    mod.set_extremes([[0.,1.],[0.,1.]])
    mod.calibrate_model(data)
    mod.plot_model()


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



