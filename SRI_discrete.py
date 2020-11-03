import numpy as np
import random
from loguru import logger
from random import randrange
import argparse
import pandas as pd

import matplotlib.pyplot as plt

'''
This is an implementation of the Susceptible, Infected, Recovered model for 02-718.

Simon Levine-Gottreich, 2020

Ex: SRI_discrete.py --m 100 --n 1000 --k 10 --lambdas 1,1,0
'''

def main():

    trial_params = {}
    trial_params['1'] = (100,1000,10,2,1,0) #(m=100,N=1000,k=10,l1_0=2,l2=1,l3=0)
    trial_params['2'] = (100,1000,10,2,1,0.001) #(m=100,N=1000,k=10,l1_0=2,l2=1,l3=0.001)
    trial_params['3'] = (100,1000,10,2,1,0.01) #(m=100,N=1000,k=10,l1_0=2,l2=1,l3=0.01)
    trial_params['4'] = (100,1000,10,2,1,0.1) #(m=100,N=1000,k=10,l1_0=2,l2=1,l3=0.1)

    for trial, params in trial_params.items():
        run_and_write_outputs(trial,params,show_plot=False)

    # args = parser.parse_args()
    # world = World(args.m,args.N,args.k,args.l1_0,args.l2,args.l3)



def run_and_write_outputs(trial,params,show_plot=False):

    world = World(*params)

    results = CTMM(world)

    for i in results:
        print(i)

    # results = [get_resultant_values(world_after_covid.countries[i]) for i in range(world_after_covid.m)]

    # r

    logger.info('Final results are:')
    results_df = pd.DataFrame.from_records(results).rename(columns={0:'lambda_i,1',1:'Ri/(Ri+Si)'})
    logger.info(results_df)
    logger.info('saved results to csv, saved plot')

    results_df.plot(title='λi1 (X) vs. final Ri/(Ri + Si) (Y) over all populations (Trial #'+trial+')',
                    kind='scatter', x='lambda_i,1',y = 'Ri/(Ri+Si)')
    if show_plot == True:
        plt.show()
    #returning file
    plt.savefig('output_'+trial+'.png')
    results_df.to_csv('results'+trial+'.csv')

class Country:
    # make one country i, for 1...i...m
    def __init__(self, N,k,l1_0,l2,l3):

        # self.N = N

        #initialize parameters
        self.S = N-k
        self.I = k
        self.R = 0        

        self.l1 = np.random.uniform(low=0.0,high=l1_0)
        self.l2 = l2
        self.l3 = l3

    @property
    def N(self):
        return self.S + self.I + self.R

def sample_from_exp(rate_param):
    return np.random.exponential((1/rate_param)) if rate_param > 0 else float('inf')

class World:
    def __init__(self, m, N, k, l1_0, l2, l3):

        self.m = m
        # initialize countries
        self.countries = [Country(N,k,l1_0,l2,l3) for _ in range(self.m)]

    @property
    def I(self):
        return np.sum([c.I for c in self.countries])

    def new_migration(self, country_i:Country, country_j:Country):
        #migrate, and assume a uniform distribution across
        # S,I,and R groups for who will migrate...

        #not the cleanest but whatever...
        subpopulations_i = [country_i.S,country_i.I,country_i.R]
        able_to_migrate_i = [p!=0 for p in subpopulations_i] #1,0,1 , for ex.

        choice_i_idx = randrange(len(able_to_migrate_i))
        choice_i = able_to_migrate_i[choice_i_idx]

        while choice_i == 0:
            # take a random person from a group that actually
            # still has people!
            del able_to_migrate_i[choice_i_idx] #remove from consideration
            choice_i_idx = randrange(len(able_to_migrate_i)) #resample
            choice_i = able_to_migrate_i[choice_i_idx] #choose new subpopulation from N = S,I, or R.

        if choice_i_idx == 0:
            country_i.S -= 1
            country_j.S += 1
        elif choice_i_idx == 1:
            country_i.I -= 1
            country_j.I += 1
        elif choice_i_idx == 2:
            country_i.R -= 1
            country_j.R += 1

    def update_sum_of_infected(self):
        self.sum_of_infected = np.sum([c.I for c in self.countries])
    

def get_resultant_values(country_i:Country) -> float:
    '''
    Returns:
    - tuple, (population l1 parameter, final fraction of infected)
    '''
    return (country_i.l1, (country_i.R/(country_i.N)) )


def get_rate_of_infection(country):
    return country.l1*country.I*(country.S/country.N)

def get_rate_of_recovery(country):
    return country.l2*country.I

def get_rate_of_migration_out(country):
    return country.l3*country.N 

def CTMM(world:World):

    '''Run a continuous time markov model based on
    an initialized World object'''

    results = []

    t=0
    # for _ in range(100):
    while world.I > 0:
        for i, country in enumerate(world.countries):

            t_i = sample_from_exp(country.l1*country.I*(country.S/country.N))
            t_r = sample_from_exp(country.l2*country.I)
            t_m = sample_from_exp(country.l3*country.N )

            # logger.info((t_i, t_r, t_m))

            if not (t_i == float('inf') and t_r == float('inf') and t_m == float('inf')):

                t_min = min(t_i,t_r,t_m)
                
                if t_i == t_min:
                    t+=t_i
                    country.I += 1
                    country.S -= 1
                    #including these staments in all conditionals in case there's a tie in the min (?)
                elif t_r == t_min:
                    # world.new_recovery(country)
                    t+=t_r
                    country.I -= 1
                    country.R += 1
                elif t_m == t_min:
                    #randomly select a destination country for migration...
                    t+=t_m
                    j = np.random.choice(np.setdiff1d(range(world.m), i))
                    world.new_migration(country,world.countries[j])

    for country in world.countries:
        results.append((country.l1, country.R/country.N))
        
    return results


        
# args = parser.parse_args()

if __name__=="__main__":
    # main(args)
    main()

