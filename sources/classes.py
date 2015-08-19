__author__ = 'eliazarinelli'

import functions as fn
import pandas as pd
import pylab as pl
from datetime import datetime


class AncernoDatabase():
    def __init__(self):
        self.ancerno_name_col = ['symbol', 'side', 'Q', 'VP', 'VD', 'sigma', 't_s', 't_e', 'p_s', 'p_e', 'pi', 'eta', 'dur', 'imp']
        self.db = pd.DataFrame(columns = self.ancerno_name_col)

    def import_dataset(self):
        self.db = pd.read_csv('../data_frame/df_07_sel.csv',delimiter=';')
        self.db['pi'] =  1. * self.db.Q/self.db.VD
        self.db['eta'] = 1. * self.db.Q/self.db.VP
        self.db['dur'] = 1. * self.db.VP/self.db.VD
        self.db['imp'] = 1. * self.db.side * pl.log10(self.db.p_e/self.db.p_s) / self.db.sigma

    def get_symbol(self):
        return self.db.symbol

    def get_side(self):
        return self.db.side

    def get_Q(self):
        return self.db.Q

    def get_VP(self):
        return self.db.VP

    def get_sigma(self):
        return self.db.sigma

    def get_t_s(self):
        return self.db.t_s

    def get_t_e(self):
        return self.db.t_e

    def get_p_s(self):
        return self.db.p_s

    def get_p_e(self):
        return self.db.p_e

    def get_eta(self):
        return self.db.eta

    def get_dur(self):
        return self.db.dur

    def get_pi(self):
        return self.db.pi

    def get_ym(self):
        return self.db.t_s.apply(fn.extract_ym)

    def get_t_s_mm(self):
        return self.db.t_s.apply(fn.extract_min)

    def get_t_s_mm_2(self):
        return map(fn.extract_min,self.db.t_s.tolist())

    def get_t_e_mm(self):
        return self.db.t_e.apply(fn.extract_min)




class Filters():
    def __init__(self, \
                 symbols = ['AIG', 'BAC', 'C', 'CSCO', 'GE', 'JPM', 'MRK', 'MSFT', 'PG', 'XOM'], \
                 months = ['2007-01', '2007-02', '2007-03', '2007-04', '2007-05', '2007-06', \
                           '2007-07', '2007-08', '2007-09', '2007-10', '2007-11', '2007-12', ], \
                 t_s = ['08:00:00','18:00:00'], \
                 t_e = ['08:00:00','18:00:00'], \
                 pi = [0.,1.], \
                 eta = [0.,1.], \
                 dur = [0.,1.]):
        self.symbols = symbols
        self.months = months
        self.extremes = {'t_s': t_s, \
                         't_e': t_e, \
                         'pi' : pi, \
                         'eta': eta, \
                         'dur': dur}

    def get_symbol(self):
        return self.symbols

    def get_ym(self):
        return self.months

    def get_t_s_mm(self):
        return map(fn.extract_min_short,self.extremes['t_s'])

    def get_t_e_mm(self):
        return map(fn.extract_min_short,self.extremes['t_e'])

    def get_pi(self):
        return self.extremes['pi']

    def get_eta(self):
        return self.extremes['eta']

    def get_dur(self):
        return self.extremes['dur']





#class Database2D():
#    def __init__(self, db_info=None):
#        self.db = pd.DataFrame(data = pl.ones((100,2)), columns = ['pi','imp'])
#        self.db_info = db_info or DatabaseInfo()



#a = Database2D()
#print a.db_info.months

class Impact2D:

    def __init__(self, n_points = 10 ):

        self.n_points = n_points
        self.data = pd.DataFrame(data = 0.1*pl.ones((self.n_points,4)), columns = ['pi','imp','stdd','nn'])
        self.functions = {'power law': fn.ff_pl, 'logarithm': fn.ff_lg}
        self.parameters = {'power law': [1., 1.], 'logarithm': [1., 1.]}
        self.errors = {'power law': [1., 1.], 'logarithm': [1., 1.]}
        self.chi = {'power law': 1., 'logarithm': 1.}

    def __str__(self):
       return '[PL: %f, %f] \n[LG: %f, %f]' % (self.parameters['power law'][0], self.parameters['power law'][1], self.parameters['logarithm'][0], self.parameters['logarithm'][1])

    def set_data(self, data_in):
        self.data = data_in
        #self.n_points = len(data_in.pi)

    def get_data(self):
        return self.data

    def set_par(self, key_in, data_in):
        self.parameters[key_in] = data_in[0:2]
        self.errors[key_in] = data_in[2:4]
        self.chi[key_in] = data_in[4]

    def get_par(self, key_in):
        return self.parameters[key_in]

    def get_err(self, key_in):
        return self.errors[key_in]

    def get_chi(self, key_in):
        return self.chi[key_in]

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
        l_02 = '$E_{RMS} = $' + str("%.4f" % round(pl.sqrt(self.get_chi('power law')/self.n_points),4))
        leg_0 = l_00 + " " + l_01 + " " + l_02

        l_10 = '$\hat{a} = $' + str("%.4f" % round(self.get_par('logarithm')[0],4)) + '$\pm$' + str("%.4f" % round(self.get_err('logarithm')[0],4))
        l_11 = '$\hat{b} = $' + str("%.4f" % round(self.get_par('logarithm')[1],4)) + '$\pm$' + str("%.4f" % round(self.get_err('logarithm')[1],4))
        l_12 = '$E_{RMS} = $' + str("%.4f" % round(pl.sqrt(self.get_chi('logarithm')/self.n_points),4))
        leg_1 = l_10 + " " + l_11 + " " + l_12
        l1 = pl.legend([p_pl,p_lg], ['$f(\phi) = Y\phi^{\delta}$', '$g(\phi)= a \log_{10}(1+b\phi)$'], loc=2, prop={'size':15})
        l2 = pl.legend([p_pl,p_lg], [leg_0 ,leg_1 ], loc=4, prop={'size':15})
        pl.gca().add_artist(l1)
        pl.subplots_adjust(bottom=0.15)
        pl.subplots_adjust(left=0.17)
        pl.show()

class Calibrator:

    def __init__(self, ancerno_database = None, filters = None, n_points = 10):
        self.database = ancerno_database or AncernoDatabase()
        self.filter = filters or Filters()
        self.n_points = n_points
        self.impact = Impact2D(n_points = self.n_points)

    def set_database(self, ancerno_database):
        self.database = ancerno_database

    def set_filters(self, filters):
        self.filter = filters

    def set_n_points(self, n_points):
        self.n_points = n_points
        self.impact = Impact2D(n_points = self.n_points)


    def calibrate(self):

        # Filtering database with respect to the parameters set in Filters
        filter_symbol = fn.is_in(self.database.get_symbol(), self.filter.get_symbol())
        filter_months = fn.is_in(self.database.get_ym(), self.filter.get_ym())
        filter_t_s_0 = self.database.get_t_s_mm() > self.filter.get_t_s_mm()[0]
        filter_t_s_1 = self.database.get_t_s_mm() < self.filter.get_t_s_mm()[1]
        filter_t_e_0 = self.database.get_t_e_mm() > self.filter.get_t_e_mm()[0]
        filter_t_e_1 = self.database.get_t_e_mm() < self.filter.get_t_e_mm()[1]
        filter_pi_0 = self.database.get_pi() > self.filter.get_pi()[0]
        filter_pi_1 = self.database.get_pi() < self.filter.get_pi()[1]
        filter_eta_0 = self.database.get_eta() > self.filter.get_eta()[0]
        filter_eta_1 = self.database.get_eta() < self.filter.get_eta()[1]
        filter_dur_0 = self.database.get_dur() > self.filter.get_dur()[0]
        filter_dur_1 = self.database.get_dur() < self.filter.get_dur()[1]

        filter_all = filter_t_s_0 & filter_t_s_1 & filter_t_e_0 & filter_t_e_1 & filter_pi_0 & filter_pi_1 \
                     & filter_eta_0 & filter_eta_1 & filter_dur_0 & filter_dur_1 & filter_symbol & filter_months

        # Extracting pi and imp
        database_fr = self.database.db.loc[filter_all,['pi','imp'] ]

        # Generating the bin extremes
        bin_end_imp_pi = pl.percentile(database_fr.pi,list(100.*pl.arange(self.n_points+1.)/(self.n_points)))

        # Adjusting the last bin extreme
        bin_end_imp_pi[-1] = bin_end_imp_pi[-1] + 0.00001

        # Assigning each point to a bin
        database_fr['fac_pi'] = pl.digitize(database_fr.pi,bin_end_imp_pi)

        # Using a groupby in order to generate average pi and imp for each bin, assigning the output to df_imp
        df_gp = database_fr[['pi','imp','fac_pi']].groupby('fac_pi')
        df_imp = df_gp.mean()

        # Putting the output to Impact2D
        self.impact.set_data(df_imp)









db = AncernoDatabase()
db.import_dataset()

a = Calibrator(n_points=30)
a.set_database(db)
a.calibrate()
a.impact.plot_impact()



#a.plot_impact()
#a = ImpactStructure2D()


