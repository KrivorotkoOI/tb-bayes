import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as copy
from model import Data, ode_objective, ode_solve, eval_grads, precompile_model
import sciris as sc

with open("tb-mbr/paramteres.yml", "r", encoding="utf8") as file:
    p = yaml.safe_load(file)

def output(string: str):
    with open("output.txt", "a") as file:
        file.write(string + "\n")


initial_state = p["initial_state"]
params = p["default_params"]

print(initial_state)
t_start = 2009
modelling_time = 10
t_end = t_start + modelling_time

step = 1/120

times_to_save = list(range(1,modelling_time+1))
steps_to_save = np.array(times_to_save)/step

model_kwargs = {
    "initial_state": initial_state,
    "t_start": t_start,
    "t_end": t_end,
    "step":step,
    "steps_to_save":steps_to_save,
} | p["model_kwargs"]

equation_strings, custom_vars = precompile_model(model_kwargs['equation_strings'], model_kwargs['custom_vars'])
model_kwargs['equation_strings'] = equation_strings
model_kwargs['custom_vars'] = custom_vars

results = ode_solve(params=params, **model_kwargs)

print(results)
print(results.data.shape)
print(results.points.shape)

#------------------------
#--------Plotting--------
#------------------------

def plotting(key):
    c='blue'
    m=''
    if 'T' in key:
        m='+'
    if 'I' in key:
        c='red'
    if 'g' in key:
        c = 'dark'+c
    plt.plot(results[key].points, results[key].data, label=key,marker=m,markevery=120,c=c)


plotting('S')
plt.show()

for key in ['EsT','Es', 'IsT', 'Is']:#key = 's'
    plotting(key)
plt.legend()
plt.show()

for key in ['EgT','Eg', 'IgT', 'Ig']:#key = 's'
    plotting(key)
plt.legend()
plt.show()


    

print(results[key].points, results[key].data)
print('solved direct')
