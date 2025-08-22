import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as copy
from model import Data, DataRagged, objective, ode_model, ode_solve, precompile_model, loss_func
from optuna_optimiser import run_optuna

import sqlite3

np.set_printoptions(precision=8)

with open("tb-mbr/paramteres.yml", "r", encoding="utf8") as file:
    p = yaml.safe_load(file)


def output(string: str):
    with open("output.txt", "a") as file:
        file.write(string + "\n")

#renaming
_temp_keys = p["model_kwargs"]["equation_strings"].keys()
params = p["default_params"]
estimation_bounds = p["estim_and_bounds"]

def custom_model(initial_state, params, **kwargs):
    coef = params['non-treated']
    initial_state["Es"] = np.array(initial_state["EsT"]) * coef
    initial_state["Eg"] = np.array(initial_state["EgT"]) * coef
    initial_state["Is"] = np.array(initial_state["IsT"]) * coef
    initial_state["Ig"] = np.array(initial_state["IgT"]) * coef
    res = ode_solve(initial_state=initial_state, params=params, **kwargs)
    res["EsT + EgT"] = Data(data=res["EsT"].data + res["EgT"].data, points=res["EgT"].points, keys=["ET"])
    #res["N"] = Data(data=(res["EsT + EgT"].data + res["IgT"].data + res["IsT"].data + res["Ig"].data + res["Is"].data 
    #                + res["Eg"].data + res["Es"].data + res["S"].data)/1000, points=res["EgT"].points, keys=["N"])
    return res

def custom_objective(**kwargs):
    return objective(model=custom_model, **kwargs)

model_kwargs = {
        #"init_x": initial_state,
        #'step': 1/100,
        #"t_start": t_start,
        #"t_end":t_end,
        #"data_ex": inverse_problem_data,
        "default_params": params,
        "objective": custom_objective,#ode_objective,
        "estimation_bounds": estimation_bounds,
        "n_workers": 10,
        "n_trials": 700,
    } | p["model_kwargs"]

#precompile equation strings for faster evaluation
equation_strings, custom_vars = precompile_model(model_kwargs['equation_strings'], model_kwargs['custom_vars'])
model_kwargs['equation_strings'] = equation_strings
model_kwargs['custom_vars'] = custom_vars

#run database with all history data
db_filename = "tuberculosis.db"
connection = sqlite3.connect(db_filename)
cursor = connection.cursor()
cursor.execute("SELECT * from 'Reg'")
all_regions = cursor.fetchall()

for region in all_regions[50:]:
    #collect needed data for current region
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
        continue
    
    coef = 0
    #prepare data for inverse problem and initial state
    try:
        df_region["IsT"] = df_region["IgT"] * (100 - df_region["IgT_part"]) / df_region["IgT_part"]
        df_region["Ig"] = df_region["IgT"] * coef
        df_region["Is"] = df_region["IsT"] * coef
        
        df_region["EsT + EgT"] = df_region['contingents'] - df_region["IsT"] - df_region["IgT"]
        df_region["EsT"] = df_region["EgT"] = df_region["EsT + EgT"]/2
        df_region["Es"] = df_region["EsT"] * coef
        df_region["Eg"] = df_region["EgT"] * coef
        
        df_region["S"] = df_region["N"]*1000 - df_region["EsT + EgT"] - df_region["IsT"] - df_region["IgT"] - df_region["Es"] - df_region["Eg"] - df_region["Is"]- df_region["Ig"] 
    except:
        continue
    df_region = df_region.drop(2019)
    points = []
    values = []
    for key in p["passed_keys"]:  # df_region.columns:
        temp_df = df_region[key].dropna()
        points.append(list(temp_df.index))
        values.append(list(temp_df))
    inverse_problem_data = DataRagged(keys=p["passed_keys"], points=points, data=values)
    params["N"] = df_region["N"].iloc[0]
    model_kwargs["data_ex"] = inverse_problem_data
    #inverse_problem_data.points==2019
    
    initial_state = dict.fromkeys(_temp_keys, np.nan)
    t_start=df_region.dropna().sort_index().index[0]
    for key in _temp_keys:
        initial_state[key] = [df_region.dropna().sort_index().iloc[0][key]]
    model_kwargs["initial_state"] = initial_state
    model_kwargs["t_start"]=t_start 
    try:
        print("Optuna run")
        optuna_dict, best_val = run_optuna(show_progress_bar=True, **model_kwargs)

        optuna_P = params.copy()
        for key in estimation_bounds.keys():
            optuna_P[key] = optuna_dict[key]

        print("Start plotting")

        
        optuna = custom_model(params=optuna_P, **model_kwargs)
        for key in ['IgT','IsT','EsT + EgT']: #,'N'
            data = optuna[key]
            plt.plot(data.points, data.data)
            data2 = optuna[key[:2]]
            plt.plot(data2.points, data2.data)
                        
            inv_data = inverse_problem_data[key]
            plt.scatter(inv_data.points, inv_data.data)
            plt.savefig(str(region[1])+key+'.pdf')
            plt.cla()
        output(str(region[1]))
        output(str(optuna_P))
        output(str(best_val))
    except:
        print('Failed region', region[1])
        continue

connection.close()
