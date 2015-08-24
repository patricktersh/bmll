__author__ = 'eliazarinelli'

import unittest
import classes as cl
import pandas as pd
import pylab as pl

class TestAncernoDatabase(unittest.TestCase):

    def test_import_dataset_csv(self):

        db_test = cl.AncernoDatabase()
        db_test.import_dataset_csv('../data_frame/test_import_database_csv.csv')
        # some tests on the correct calculation of derivate quantities: pi, eta, dur, imp
        self.assertEqual(db_test.db_raw['pi'][0], 1./3.)
        self.assertEqual(db_test.db_raw['eta'][0], 1./2.)
        self.assertEqual(db_test.db_raw['dur'][0], 2./3.)
        self.assertEqual(round(db_test.db_raw['imp'][0],4), 230.2585)

    def test_apply_filter(self):
        # Setting a test filter
        fil_test = cl.Filter()
        fil_test.set_symbols(['AAPL'])
        fil_test.set_months(['2007-01'])
        fil_test.set_t_s_min('08:30:00')
        fil_test.set_t_s_max('09:30:00')
        fil_test.set_t_e_min('12:30:00')
        fil_test.set_t_e_max('13:30:00')
        fil_test.set_pi_min(0.0001)
        fil_test.set_pi_max(0.1)
        fil_test.set_eta_min(0.001)
        fil_test.set_eta_max(1.)
        fil_test.set_dur_min(0.09)
        fil_test.set_dur_max(0.11)

        # Reading an Ancerno Dataset
        db_test = cl.AncernoDatabase()
        db_test.import_dataset_csv('../data_frame/test_apply_filter_csv.csv')
        db_test.apply_filter(fil_test)
        # Import an AncernoDatabase with 107 rows, the applied filter is such that filters out the last 7 rows
        self.assertEqual(len(db_test.get_symbol()),100)

    def test_get_binned_data_2d(self):
        # Setting a test filter
        fil_test = cl.Filter()
        fil_test.set_symbols(['AAPL'])
        fil_test.set_months(['2007-01'])
        fil_test.set_t_s_min('08:30:00')
        fil_test.set_t_s_max('09:30:00')
        fil_test.set_t_e_min('12:30:00')
        fil_test.set_t_e_max('13:30:00')
        fil_test.set_pi_min(0.0001)
        fil_test.set_pi_max(0.1)
        fil_test.set_eta_min(0.001)
        fil_test.set_eta_max(1.)
        fil_test.set_dur_min(0.09)
        fil_test.set_dur_max(0.11)

        # Reading an Ancerno Dataset
        db_test = cl.AncernoDatabase()
        db_test.import_dataset_csv('../data_frame/test_apply_filter_csv.csv')
        db_test.apply_filter(fil_test)
        a = db_test.get_binned_data_2d(10)
        self.assertEqual(a.get_source(),'../data_frame/test_apply_filter_csv.csv')
        self.assertEqual(a.get_n_bins(),10)
        self.assertEqual(a.get_filter().extremes['t_s'][0],'08:30:00')
        self.assertEqual(round(a.get_data().loc[5,'imp'],4),4.3965)

def f_lin(x,a,b):
    return a+b*x

class TestImpactModel2D(unittest.TestCase):

    def test_calibrate_model(self):
        a = cl.BinnedData2D()
        #'pi','imp','stdd','nn'
        a.data.pi = pl.linspace(1.,10.,10)
        tmp = pl.linspace(1.,10.,10)
        tmp[1] = 1.1
        tmp[4] = 3.9
        a.data.imp = tmp

        b = cl.ImpactModel2D(n_parameters=2)
        b.set_model(f_lin)
        b.set_extremes([[-5,5],[-5,5]])
        b.calibrate_model(a)
        self.assertEqual(round(b.get_parameters()[0],4),-0.4467)
        self.assertEqual(round(b.get_parameters()[1],4),1.0448)
        self.assertEqual(round(b.get_errors()[0],4),0.0848)
        self.assertEqual(round(b.get_errors()[1],4),0.0022)
        self.assertEqual(round(b.get_chi(),4),1.4541)

if __name__ == '__main__':
    unittest.main()