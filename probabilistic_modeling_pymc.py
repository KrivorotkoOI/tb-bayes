import json
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import arviz as az
import sqlite3
import pandas as pd
import yaml
from pathlib import Path

from model import Data, DataRagged, objective, ode_model, ode_solve, precompile_model, loss_func

with open("tb-mbr/paramteres.yml", "r", encoding="utf8") as file:
    p = yaml.safe_load(file)

@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    params['beta'] = theta[0]
    params['gamma_1'] = theta[1]
    params['gamma_2'] = theta[2]
    params['omega'] = theta[3]
    params['phi_m'] = theta[4]
    params['phi_p'] = theta[5]
    params['tg'] = theta[6]
    params['alpha'] = theta[7]
    params['non-treated-minus'] = theta[8]
    params['non-treated-plus'] = theta[9]

    coef_minus = params['non-treated-minus']
    coef_plus = params['non-treated-plus']
    initial_state["Es"] = initial_state["EsT"] * coef_minus
    initial_state["Eg"] = initial_state["EgT"] * coef_minus
    initial_state["Is"] = initial_state["IsT"] * coef_plus
    initial_state["Ig"] = initial_state["IgT"] * coef_plus

    res = ode_solve(params=params, initial_state=initial_state, **model_kwargs)
    res["EsT + EgT"] = Data(data=res["EsT"].data + res["EgT"].data, points=res["EgT"].points, keys=["ET"])

    data = {
        'index': res['IgT'].points.reshape(-1),
        'IsT': res['IsT'].data.reshape(-1),
        'IgT': res['IgT'].data.reshape(-1),
        'EsT + EgT': res['EsT + EgT'].data.reshape(-1)
    }

    return pd.DataFrame(data).set_index('index').drop(2019).to_numpy()
    
#run database with all history data
db_filename = "tuberculosis.db"
connection = sqlite3.connect(db_filename)
cursor = connection.cursor()
cursor.execute("SELECT * from 'Reg'")
all_regions = cursor.fetchall()

# (81, 'Новосибирская область')
#region = all_regions[81:82][0]
# (75, 'Республика Тыва')
#region = all_regions[75:76][0]
# (74, 'Республика Алтай')
region = all_regions[74:75][0]
# (74, 'Республика Алтай')
#region = all_regions[79:80][0]
# (87, 'Забайкальский край')
#region = all_regions[87:88][0]
print(region)

cursor.execute(
    "SELECT Year, ParameterID, value from Data WHERE RegionID = " + str(region[0])
)
reg_data = cursor.fetchall()

cursor.execute("SELECT * from Par")
all_pars = dict(cursor.fetchall())
try:
    df_region = pd.DataFrame(columns=list(all_pars.values()))
    for date, parameterID, value in reg_data:
        try:
            df_region.loc[date, parameterID] = np.float32(value)
        except ValueError:
            df_region.loc[date, parameterID] = np.nan
    params_from_db = list(p["db_keys"].values())
    df_region = df_region[params_from_db]
    df_region.columns = list(p["db_keys"].keys())
    print('All requested data is present for region ' + str(region))

except KeyError:
    print('There is missing data for chosen region ' + str(region))
    
coef = 0
# prepare data for inverse problem and initial state
try:
    df_region["IsT"] = df_region["IgT"] * (100 - df_region["IgT_part"]) / df_region["IgT_part"]
    df_region["Ig"] = df_region["IgT"] * coef
    df_region["Is"] = df_region["IsT"] * coef

    df_region["EsT + EgT"] = df_region['contingents'] - df_region["IsT"] - df_region["IgT"]
    df_region["EsT"] = df_region["EgT"] = df_region["EsT + EgT"] / 2
    df_region["Es"] = df_region["EsT"] * coef
    df_region["Eg"] = df_region["EgT"] * coef

    df_region["S"] = pd.eval("N*1000-EsT-EgT-IsT-IgT-Es-Eg-Is-Ig", global_dict=df_region)
except:
    print('prepare data error')
    
df_region = df_region.sort_index()
df_region = df_region.drop(2019)
points = []
values = []
for key in p["passed_keys"]:  # df_region.columns:
    temp_df = df_region[key].dropna()
    points.append(list(temp_df.index))
    values.append(list(temp_df))

print(p["passed_keys"])
print(points)
print(values)

print(df_region[['IsT', 'IgT', 'EsT + EgT']])


# Initial State

_temp_keys = p["model_kwargs"]["equation_strings"].keys()

initial_state = dict.fromkeys(_temp_keys, np.nan)
for key in _temp_keys:
    initial_state[key] = df_region.dropna().sort_index().iloc[0][key]
    
t_start=df_region.dropna().sort_index().index[0]
t_end=df_region.dropna().sort_index().index[-1] # 2028
step = 1

model_kwargs = {
} | p["model_kwargs"]
equation_strings, custom_vars = precompile_model(model_kwargs['equation_strings'], model_kwargs['custom_vars'])
model_kwargs['equation_strings'] = equation_strings
model_kwargs['custom_vars'] = custom_vars

model_kwargs["t_start"] = t_start
model_kwargs["t_end"] = t_end
model_kwargs["step"] = step


# Observed Data

df_region = df_region[['IsT', 'IgT', 'EsT + EgT']].loc[t_start: t_end]

# Region initial parameters (Optuna)

## Novosibirsk region
# params = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'N': 2662.395}
# q = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'beta': 2.654346520660705, 'gamma_1': 0.3469786655125636, 
#      'gamma_2': 0.505749789891934, 'omega': 2.071221852675745, 'phi_m': 0.23507541421386322, 
#      'phi_p': 2.256611292357248, 'tg': 0.05065008936871193, 'alpha': 0.1601592462668639, 
#      'non-treated-minus': 0.4861400303057526, 'non-treated-plus': 0.35143218570671464, 'N': 2662.395}

## Republic of Tyva
# params = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'N': 306.688}
# q = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'beta': 2.601793732527799, 'gamma_1': 0.319941331626464, 
#      'gamma_2': 0.8006691263019727, 'omega': 3.6202351047926182, 'phi_m': 0.2098934922097551, 
#      'phi_p': 2.4284100965259787, 'tg': 0.03402143708036411, 'alpha': 0.09155680517496287, 
#      'non-treated-minus': 0.47741062816841157, 'non-treated-plus': 0.34772439562448426, 'N': 306.688}

## Republic of Altay 
params = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'N': 205.641}
q = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'beta': 4.327767129197541, 'gamma_1': 0.26013643859076974, 
     'gamma_2': 0.5372545052276511, 'omega': 0.6285472742679823, 'phi_m': 0.3245256229587285, 
     'phi_p': 2.839711394080422, 'tg': 0.04062440431740133, 'alpha': 0.04827864464936903, 
     'non-treated-minus': 0.4494758508276996, 'non-treated-plus': 0.3423880868783986, 'N': 205.641}

## Irkutsk region
#params = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'N': 2429.757}
# q = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'beta': 2.4890246346229015, 'gamma_1': 0.2847437061544325, 
#      'gamma_2': 0.3246048091657131, 'omega': 1.216284773894436, 'phi_m': 0.2322545912321422, 
#      'phi_p': 2.1116876911098963, 'tg': 0.001857910804505892, 'alpha': 0.07885276264567517, 
#      'non-treated-minus': 0.3359328371404097, 'non-treated-plus': 0.3623936751468885, 'N': 2429.757}

## Zabaikalsky region
#params = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'N': 1106.977}
# q = {'b': 0.0142, 'mu': 0.0174, 'k': 4000000, 'beta': 2.4604959237358273, 'gamma_1': 0.41880543161678996, 
#      'gamma_2': 0.3316293220208531, 'omega': 0.8507734314463249, 'phi_m': 0.3437445535486679, 
#      'phi_p': 1.707315405011937, 'tg': 0.04916855132926207, 'alpha': 0.09035720280784947, 
#      'non-treated-minus': 0.4910789037827716, 'non-treated-plus': 0.22735980734715935, 'N': 1106.977}

def direct_problem(theta):
    params['beta'] = theta[0]
    params['gamma_1'] = theta[1]
    params['gamma_2'] = theta[2]
    params['omega'] = theta[3]
    params['phi_m'] = theta[4]
    params['phi_p'] = theta[5]
    params['tg'] = theta[6]
    params['alpha'] = theta[7]
    params['non-treated-minus'] = theta[8]
    params['non-treated-plus'] = theta[9]

    coef_minus = params['non-treated-minus']
    coef_plus = params['non-treated-plus']
    initial_state["Es"] = initial_state["EsT"] * coef_minus
    initial_state["Eg"] = initial_state["EgT"] * coef_minus
    initial_state["Is"] = initial_state["IsT"] * coef_plus
    initial_state["Ig"] = initial_state["IgT"] * coef_plus

    res = ode_solve(params=params, initial_state=initial_state, **model_kwargs)
    res["EsT + EgT"] = Data(data=res["EsT"].data + res["EgT"].data, points=res["EgT"].points, keys=["ET"])

    data = {
        'index': res['IgT'].points.reshape(-1),
        'IsT': res['IsT'].data.reshape(-1),
        'IgT': res['IgT'].data.reshape(-1),
        'EsT + EgT': res['EsT + EgT'].data.reshape(-1)
    }

    return pd.DataFrame(data).set_index('index').drop(2019)
    
p = {
    'beta': q['beta'],
    'gamma_1': q['gamma_1'],
    'gamma_2': q['gamma_2'],
    'omega': q['omega'],
    'phi_m': q['phi_m'],
    'phi_p': q['phi_p'],
    'tg': q['tg'],
    'alpha': q['alpha'],
    'non-treated-minus': q['non-treated-minus'],
    'non-treated-plus': q['non-treated-plus']
}

res = direct_problem(list(p.values()))
for key in df_region.columns:
    print(key)
    df_region[key].plot()
    res[key].plot()
    plt.show()
    

# PyMC
@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    params['beta'] = theta[0]
    params['gamma_1'] = theta[1]
    params['gamma_2'] = theta[2]
    params['omega'] = theta[3]
    params['phi_m'] = theta[4]
    params['phi_p'] = theta[5]
    params['tg'] = theta[6]
    params['alpha'] = theta[7]
    params['non-treated-minus'] = theta[8]
    params['non-treated-plus'] = theta[9]

    coef_minus = params['non-treated-minus']
    coef_plus = params['non-treated-plus']
    initial_state["Es"] = initial_state["EsT"] * coef_minus
    initial_state["Eg"] = initial_state["EgT"] * coef_minus
    initial_state["Is"] = initial_state["IsT"] * coef_plus
    initial_state["Ig"] = initial_state["IgT"] * coef_plus

    res = ode_solve(params=params, initial_state=initial_state, **model_kwargs)
    res["EsT + EgT"] = Data(data=res["EsT"].data + res["EgT"].data, points=res["EgT"].points, keys=["ET"])

    data = {
        'index': res['IgT'].points.reshape(-1),
        'IsT': res['IsT'].data.reshape(-1),
        'IgT': res['IgT'].data.reshape(-1),
        'EsT + EgT': res['EsT + EgT'].data.reshape(-1)
    }

    return pd.DataFrame(data).set_index('index').drop(2019).to_numpy()
    
with pm.Model() as tb_model:
    # Priors
    beta = pm.TruncatedNormal("beta", mu=q['beta'], sigma=2, lower=0.001, upper=20, initval=q['beta'])
    tg = pm.TruncatedNormal("tg", mu=q['tg'], sigma=0.05, lower=0.001, upper=0.5, initval=q['tg'])
    omega = pm.TruncatedNormal("omega", mu=q['omega'], sigma=0.4, lower=0.001, upper=4, initval=q['omega'])
    alpha = pm.TruncatedNormal("alpha", mu=q['alpha'], sigma=0.05, lower=0.001, upper=0.5, initval=q['alpha'])
    phi_m = pm.TruncatedNormal("phi_m", mu=q['phi_m'], sigma=0.1, lower=0.001, upper=0.999, initval=q['phi_m'])
    phi_p = pm.TruncatedNormal("phi_p", mu=q['phi_p'], sigma=0.3, lower=0.001, upper=3, initval=q['phi_p'])
    gamma_1 = pm.TruncatedNormal("gamma_1", mu=q['gamma_1'], sigma=0.1, lower=0.001, upper=0.999, initval=q['gamma_1'])
    gamma_2 = pm.TruncatedNormal("gamma_2", mu=q['gamma_2'], sigma=0.2, lower=0.001, upper=2, initval=q['gamma_2'])
    non_treated_minus = pm.TruncatedNormal("non_treated_minus", mu=q['non-treated-minus'], sigma=0.05, lower=0.001, upper=0.5, initval=q['non-treated-minus'])
    non_treated_plus = pm.TruncatedNormal("non_treated_plus", mu=q['non-treated-plus'], sigma=0.05, lower=0.001, upper=0.5, initval=q['non-treated-plus'])
    
    sigma = pm.HalfNormal("sigma", sigma=50)

    # Ode solution function
    ode_solution = pytensor_forward_model_matrix(
        pm.math.stack([beta, gamma_1, gamma_2, omega, phi_m, phi_p,
                       tg, alpha, non_treated_minus, non_treated_plus])
    )

    # Likelihood
    pm.Normal("F_obs", mu=ode_solution, sigma=sigma, observed=df_region.to_numpy())

pm.model_to_graphviz(model=tb_model)

RESULTS_PATH = Path('results')
if not RESULTS_PATH.exists():
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Run MCMC sampling
with tb_model:
    trace = pm.sample(tune=1000,
                      draws=50000,
                      step=pm.DEMetropolis(),
                      chains=4,
                      cores=1,
                      progressbar=True)
                      
trace.to_netcdf(RESULTS_PATH / 'Altay_trace_DEMetropolis_50000.nc')

# Plot result parameters distribution

def plot_traces(traces, burnin=2000):
    summary = az.summary(traces)['mean'].to_dict()
    ax = az.plot_trace(traces,
                       backend_kwargs={"figsize": (10, 30), "layout": "constrained"})

    for i, mn in enumerate(summary.values()):
        ax[i, 0].annotate(f'{mn:.5f}', xy=(mn, 0),
                          xycoords='data', xytext=(5, 10),
                          textcoords='offset points',
                          rotation=90, va='bottom',
                          fontsize='large',
                          color='#AA0022')
                          
plot_traces(trace, burnin=0)
plt.savefig(RESULTS_PATH / "Altay_trace_DEMetropolis_50000.png")

# R-hat values

rhat = pm.rhat(trace)
print("R-hat values:")
for param, value in rhat.items():
    print(f"{param}: {value:.3f}")
    
def plot_data(ax, col_id, lw=2, title=""):
    ax.plot(df_region.index, df_region.values[:, col_id], color="b", lw=lw, marker="o", markersize=12, label=df_region.columns[col_id])
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("", fontsize=14)
    return ax

def plot_model(
    ax,
    x_y,
    col_id,
    time=np.arange(t_start, t_end, 1/120),
    alpha=1,
    lw=3,
    title="",
):
    ax.plot(time, x_y[:, col_id], color="r", alpha=alpha, lw=lw, label=df_region.columns[col_id])
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    return ax

def plot_model_trace(ax, col_id, trace_df, row_idx, lw=1, alpha=0.2):
    cols = ['beta', 'gamma_1', 'gamma_2', 'omega', 'phi_m', 'phi_p',
            'tg', 'alpha', 'non_treated_minus', 'non_treated_plus']
    row = trace_df.iloc[row_idx, :][cols].values
    theta = row
    x_y = direct_problem(theta).to_numpy()
    
    plot_model(ax, x_y, col_id, lw=lw, alpha=alpha);

def plot_inference(
    ax,
    col_id,
    trace,
    num_samples=25,
    title="",
    plot_model_kwargs=dict(lw=1, alpha=0.2),
):
    trace_df = az.extract(trace, num_samples=num_samples).to_dataframe()
    plot_data(ax, col_id, lw=0)
    for row_idx in range(num_samples):
        plot_model_trace(ax, col_id, trace_df, row_idx, **plot_model_kwargs)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    
model_kwargs['step'] = 1/120
model_kwargs['t_end'] = 2021


# Plot results for region

fig, ax = plt.subplots(figsize=(8, 6))
for col_id in range(3):
    plot_inference(ax, col_id, trace, num_samples=100, title=df_region.columns[col_id])
    plt.show()
    
fig, ax = plt.subplots(figsize=(8, 6))
plot_inference(ax, 1, trace, num_samples=100, title=df_region.columns[1])

fig, ax = plt.subplots(figsize=(8, 6))
plot_inference(ax, 2, trace, num_samples=100, title=df_region.columns[2])

# Print parameter mean and 95% CI

var_names = list(trace.posterior.data_vars)

for var in var_names:
    samples = trace.posterior[var].values.flatten()
    
    median = np.median(samples)
    lower = np.percentile(samples, 2.5)
    upper = np.percentile(samples, 97.5)
    
    print(f"{var}:")
    print(f"  Mean: {median:.3f}")
    print(f"  95% CI: [{lower:.3f}, {upper:.3f}]")