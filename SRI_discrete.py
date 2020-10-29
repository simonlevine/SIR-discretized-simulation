import numpy as np
from logger import loguru
from random import randrange
import argparse

'''
This is an implementation of the Susceptible, Infected, Recovered model for 02-718.

Simon Levine-Gottreich, 2020

Ex: SRI_discrete.py --m 100 --n 1000 --k 10 --lambdas 1,1,0
'''

def main(args):
    args = parser.parse_args()
    simulate(args)


class World:
    def __init__(self, m, N, k, l1_0, l2, l3):

        self.m = m
        # initialize countries
        self.countries = [Country(N,k,l_1_0,l2,l3) for _ in range(self.m)]

    
    def new_infection(country_i:Country):
        country_i.new_infection()
    def new_recovery(country_i: Country):
        country_i.new_recovery()

    def new_migration(country_i:Country, country_j:Country):
        #migrate, and assume a uniform distribution across
        # S,I,and R groups for who will migrate...

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

    

class Country:
    # make one country i, for 1...i...m
    def __init__(self, N,k,l_1_0,l_2,l3)

        self.N = N

        self.S = N-k
        self.I = k
        self.R = 0

        self.l1 = np.random.uniform(low=0.0,high=l_1_0)
        self.l2 = l2
        self.l3 = l3

    def new_infection():
        self.S -= 1
        self.I += 1

    def new_recovery():
        self.I -= 1
        self.R += 1




parser = argparse.ArgumentParser(
    description='Process (m, N, k, l1, l2, l3) args for CTMM (discrete SRI).'
    )

parser.add_argument(
        "--m",
        default=100,
        type=int,
        )

parser.add_argument(
        "--N",
        default=1000,
        type=float,
        )

parser.add_argument(
        "--k",
        default=10,
        type=int,
        )

parser.add_argument(
        "--l1",
        default=2.,
        type=float,
        )

parser.add_argument(
        "--l2",
        default=1.,
        type=float,
        )

parser.add_argument(
        "--l3",
        default=0.,
        type=float,
        )
        
        
args = parser.parse_args()

if __name__=="__main__":
    main(args)

