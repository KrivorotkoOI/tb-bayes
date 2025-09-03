import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
import SALib as sa
import yaml
from copy import deepcopy as copy
from model import Data, ode_solve, rungekutta4, precompile_model

with open("tb-mbr/paramteres.yml", "r", encoding="utf8") as file:
    p = yaml.safe_load(file)

def output(string: str):
    with open("output.txt", "a") as file:
        file.write(string + "\n")

initial_state = p["initial_state"]
params = p["default_params"]

t_start = 2009
modelling_time = 10
t_end = t_start + modelling_time

step = 1/200

n = int(modelling_time/step)

n_sobol = 2**8

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
model_kwargs['custom_vars']=custom_vars
model_kwargs['equation_strings']=equation_strings


p_bounds = p['estim_and_bounds']
names = list(p_bounds.keys())
bounds = list(p_bounds.values())

d=len(names)

for key in initial_state.keys():
    initial_state[key] = np.array([initial_state[key]]*int(n_sobol*(d*2+2)))
model_kwargs['initial_state'] = initial_state

compartment = 'Es'

def func(x,params=params,solver=ode_solve,model_kwargs=model_kwargs):
    for i, name in enumerate(names):
        params[name] = x[:,i] 
    result = solver(params=params, solver=rungekutta4, **model_kwargs)[compartment].data
    return result.T


from SALib import ProblemSpec
sp = ProblemSpec({
    'names': names,
    'bounds': bounds
})

import time
start_time = time.time()

(
    sp.sample_sobol(n_sobol)
    .evaluate(func)
    .analyze_sobol(print_to_console=True,
                   num_resamples=100)
)
print(time.time() - start_time)



x = np.linspace(0,modelling_time,len(steps_to_save))
# Get first order sensitivities for all outputs
S1s = np.array([sp.analysis[_y]['S1'] for _y in sp['outputs']])

# Get model outputs
y = sp.results

# Set up figure
fig = plt.figure(figsize=(12, 4), constrained_layout=True)
gs = fig.add_gridspec(1,2)

ax0 = fig.add_subplot(gs[:,0])
ax1 = fig.add_subplot(gs[:,1])
#ax2 = fig.add_subplot(gs[1, 1])

# Populate figure subplots
for i, ax in enumerate([ax1]):
    for j, name in enumerate(names):
        ax.plot(x, S1s[:, j],
            label=name)
    ax.set_xlabel("x")
    ax.set_ylabel("First-order Sobol index")

    ax.set_ylim(0, 1.04)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.legend(loc='upper right')

ax0.plot(x, np.mean(y, axis=0), label="Mean", color='black')

# in percent
prediction_interval = 95

ax0.fill_between(x,
                 np.percentile(y, 50 - prediction_interval/2., axis=0),
                 np.percentile(y, 50 + prediction_interval/2., axis=0),
                 alpha=0.5, color='black',
                 label=f"{prediction_interval} % prediction interval")

ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.legend(loc='upper center')._legend_box.align = "left"
plt.savefig(compartment+'.pdf')
plt.show()