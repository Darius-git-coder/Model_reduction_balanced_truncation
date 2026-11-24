"""Independent numerical checks for the report-specific calculations."""

import csv
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp,quad

from experiments import nonuniform_system,fdm_2d_poisson,exact_discrete
from pilot_reference import pde_reference
from adaptive_time import adaptive_euler


class ReportTests(unittest.TestCase):
    def test_source_coefficients(self):
        for a in [1.5,4/3]:
            for k in [1,2,5]:
                integral=2*quad(lambda x:np.sin(a*np.pi*x)*np.sin(k*np.pi*x),0,1)[0]
                assert_allclose(integral,np.sinc(a-k)-np.sinc(a+k),atol=1e-13)

    def test_zero_boundary_and_initial_condition(self):
        grid=np.linspace(0,1,19)
        ref=pde_reference(grid,grid,[0.,.3])
        assert_allclose(ref[:,:,0],0,atol=1e-14)
        assert_allclose(ref[[0,-1]],0,atol=1e-14)
        assert_allclose(ref[:,[0,-1]],0,atol=1e-14)

    def test_exact_discrete_solution(self):
        A,B,*_=fdm_2d_poisson(3)
        times=np.linspace(0,1,21)
        computed=solve_ivp(lambda t,x:A@x+B[:,0]*10*np.sin(2*np.pi*t),[0,1],np.zeros(9),
                           method='DOP853',rtol=1e-11,atol=1e-13,t_eval=times).y
        assert_allclose(computed,exact_discrete(3,times),atol=1e-10)

    def test_nonuniform_operator_matches_uniform(self):
        grid=np.linspace(0,1,7)
        A,B=nonuniform_system(grid,grid)
        A0,B0,*_=fdm_2d_poisson(5)
        assert_allclose(A.toarray(),A0.toarray(),atol=1e-12)
        assert_allclose(B,B0,atol=1e-13)

    def test_time_counts_and_rejections(self):
        result=adaptive_euler(np.array([[-2.]]),np.ones((1,1)),None,rtol=1e-3,atol=1e-5)
        trials=result['accepted']+result['rejected']
        self.assertEqual(result['solves'],3*trials)
        self.assertEqual(result['input_evals'],2*trials)
        self.assertEqual(len(result['times']),2*result['accepted']+1)
        self.assertGreater(result['rejected'],0)
        self.assertAlmostEqual(result['times'][-1],1.)

    def test_saved_bounds_and_errors(self):
        with open('report/data/bounds_adaptive.csv') as file:
            for row in csv.DictReader(file):
                self.assertLessEqual(float(row['estimate']),float(row['tolerance']))
                self.assertLessEqual(float(row['error']),float(row['estimate']))
        with open('report/data/rank_study.csv') as file:
            for row in csv.DictReader(file):
                self.assertLessEqual(float(row['sampled_hinf']),float(row['tail_bound'])*1.001)


if __name__=='__main__':unittest.main()
