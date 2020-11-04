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
    np.random.seed
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

    def update_sum_of_infected(self):
        self.sum_of_infected = np.sum([c.I for c in self.countries])
    

def get_resultant_values(country_i:Country) -> float:
    '''
    Returns:
    - tuple, (population l1 parameter, final fraction of infected)
    '''
    return (country_i.l1, (country_i.R/(country_i.N)) )


def CTMM(world:World):

    '''Run a continuous time markov model based on
    an initialized World object'''

    results = []

    t=0
    # for _ in range(100):
    while world.I > 0:

        infect_vec = [sample_from_exp(country.l1*country.I*(country.S/country.N)) for country in world.countries]
        t_i, i = min(infect_vec), np.argmin(infect_vec)

        recover_vec = [sample_from_exp(country.l2*country.I) for country in world.countries]
        t_r, r = min(recover_vec), np.argmin(recover_vec)

        migrateS_vec = [sample_from_exp(country.l3*country.S) for country in world.countries]
        t_m1, m1 = min(migrateS_vec), np.argmin(migrateS_vec)

        migrateI_vec = [sample_from_exp(country.l3*country.I) for country in world.countries]
        t_m2, m2 = min(migrateI_vec), np.argmin(migrateI_vec)

        migrateR_vec = [sample_from_exp(country.l3*country.R) for country in world.countries]
        t_m3, m3 = min(migrateR_vec), np.argmin(migrateR_vec)

        t_min_global = min(t_i,t_r,t_m1,t_m2,t_m3)

            # logger.info((t_i, t_r, t_m))

        if not (
            t_i == float('inf') and
            t_r == float('inf') and
            t_m1 == float('inf') and 
            t_m2 == float('inf') and
            t_m3 == float('inf')
            ):
            
            if t_i == t_min_global:
                t+=t_i
                world.countries[i].I += 1
                world.countries[i].S -= 1
                #including these staments in all conditionals in case there's a tie in the min (?)
            elif t_r == t_min_global:
                # world.new_recovery(country)
                t+=t_r
                world.countries[r].I -= 1
                world.countries[r].R += 1

            elif t_m1 == t_min_global:
                #randomly select a destination country for migration...
                t+=t_m1
                j = np.random.choice(np.setdiff1d(range(world.m), i))
                world.countries[m1].S-=1
                world.countries[j].S +=1

            elif t_m2 == t_min_global:
                #randomly select a destination country for migration...
                t+=t_m2
                j = np.random.choice(np.setdiff1d(range(world.m), i))
                world.countries[m2].I-=1
                world.countries[j].I +=1

            elif t_m3 == t_min_global:
                t+=t_m3
                j = np.random.choice(np.setdiff1d(range(world.m), i))
                world.countries[m3].R -=1
                world.countries[j].R +=1

    for country in world.countries:
        results.append((country.l1, country.R/country.N))
        
    return results


        
# args = parser.parse_args()

if __name__=="__main__":
    # main(args)
    main()

