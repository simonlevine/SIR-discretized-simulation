This is an implementation of the Susceptible, Infected, Recovered model for 02-718.

We will use a simple SIR model here to examine issues of how one might slow COVID-19 infections.

A Continuous Time Markov Model is implemented, such that events occur with X ~ Exp(λi) for event i.

- We are given the simple assumption of N people, split into 3 populations (S,I,and R).
- No reinfection.
- Each infected person contacts λ1 persons per unit time
- Of λ1 persons, S/N are infectious.
- --> Total rate of new people infected per unit time is λ1*I*(S/N)
- infected people recover at λ2, so overall recovery rate is λ2*I.

