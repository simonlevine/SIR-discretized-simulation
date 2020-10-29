This is an implementation of the Susceptible, Infected, Recovered model for infectious disease.

We will use a simple SIR model here to examine issues of how one might slow COVID-19 infections.
A Continuous Time Markov Model is implemented, such that events occur with $X \sim Exp(\lambda_i)$ for event $i$.

Simplest Case:

- We are given the simple assumption of N people, split into 3 populations $(S,I,$ and $R)$.
- No reinfection.
- Each infected person contacts $\lambda_1$ persons per unit time
- Of $\lambda_1$ persons, $S/N$ are infectious.
- --> Total rate of new people infected per unit time is λ1*I*(S/N)
- infected people recover at $\lambda_2$, so overall recovery rate is $\lambda_2I$.

General Case:

- People are from different countries affecting $\lambda_1$ differnently, with different ways of controlling immigration.
- Assume $m$ countries each with population $N_i = |S_i| + |I_i| + |R_i|$, with
  - infections at rate $\lambda_{i1}Ii\frac{S_i}{R_i}$
  - recovery at rate $\lambda_{i2}I_i$
  - Also, assume inter-country migration at $\lambda_{i3}$, for a rate of movement fooor countries $i \to j$ at $\lambda_{i3}N_i$

We will vary parameters $\lambda$ to understand how this Markov Model behaves.

Implementation details:
-Inputs:
    - $m,N,k,\lambda_1^*,\lambda_2,\lambda_3$
- 