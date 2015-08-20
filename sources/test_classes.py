__author__ = 'eliazarinelli'

import unittest
import classes as cl
import pandas as pd
import pylab as pl

class TestAncernoDatabase(unittest.TestCase):

    def test_import_dataset_csv(self):
        a = cl.AncernoDatabase()
        a.import_dataset_csv('../data_frame/test_dataframe.csv')
        # some tests on the correct calculation of derivate quantities: pi, eta, dur, imp
        self.assertEqual(a.get_pi()[0], 1./3.)
        self.assertEqual(a.get_eta()[0], 1./2.)
        self.assertEqual(a.get_dur()[0], 2./3.)
        self.assertEqual(round(a.get_imp()[0],4), 230.2585)

class TestImpact2D(unittest.TestCase):

    def test_calibrate_impact(self):

        # creating an Ancerno database
        db_test = cl.AncernoDatabase()
        db_test.import_dataset_csv('../data_frame/test_impact.csv')

        # creating the standard filter
        fil_test = cl.Filters()

        # creating an impact
        imp_test = cl.Impact2D()

        # setting the filter to the standard filter
        imp_test.set_filter(fil_test)

        # setting the number of bins for calibrating the impact
        n_bins_test = 20

        # calibrating the impact
        imp_test.calibrate_impact(db_test,n_bins_test)

        # testing the output of the calibration procedure, i.e. the best fitting parameters
        self.assertEqual(round(imp_test.get_par('power law')[0],4),0.6341)
        self.assertEqual(round(imp_test.get_par('power law')[1],4),0.6007)
        self.assertEqual(round(imp_test.get_par('logarithm')[0],4),0.0409)
        self.assertEqual(round(imp_test.get_par('logarithm')[1],4),914.2394)













if __name__ == '__main__':
    unittest.main()