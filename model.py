import numpy as np
import copy as copy
import sciris as sc
from dataclasses import dataclass
import warnings
from scipy.integrate import odeint

@dataclass
class DataRagged:
    keys: np.ndarray  # list
    points: np.ndarray  # list
    data: np.ndarray
        
    def __getitem__(self, key):
        if isinstance(key,str):
            ind = sc.findinds(self.keys, key)[0]
            return DataRagged(keys=key, points=self.points[ind], data=self.data[ind])
        else:
            return DataRagged(keys=self.keys[key], points=self.points[key], data=self.data[key])

@dataclass
class Data:
    keys: np.ndarray  # list
    points: np.ndarray  # list
    data: np.ndarray
    
    def __setitem__(self, key, value):
        self.keys=np.append(self.keys,key)
        self.points=np.append(self.points,value.points.reshape(1,*value.points.shape), axis=0)
        vds = value.data.shape
        self.data=np.append(self.data,value.data.reshape((vds[0],1,*vds[1:])), axis=1)
        
        
    def __getitem__(self, key):
        if isinstance(key,str):
            ind = sc.findinds(self.keys, key)[0]
            return Data(keys=key, points=self.points[ind], data=self.data[:,ind])
        else:
            return Data(keys=self.keys[key], points=self.points[key], data=self.data[:,key])


def scipy_solver(func, init_x: dict, t_end: float, step: float, t_start=0, solver_args=(), func_args=()):
    ## input :
    # func(parameters, x_values) - function discribes the model
    # param - system parameters

    # init_x - initial system values
    # t_start - time of start, where initial is set
    # t_end - ending
    # step - grid step
    ## output :
    # result - system vaue on grid of time points
    # result - system vaue on grid of time points
    T = t_end - t_start
    Nt = int(T / step)
    if Nt<0:
        step = -step
        Nt = -Nt
        warnings.warn('ODE INT: Check the step direction and start/end times. Automaticaly reversed step direction.')
        #raise Error('Wrong step direction')
    val_temp = np.array(list(init_x.values()))
    result = odeint(func, val_temp, np.linspace(t_start, t_end, Nt), args=func_args)
    return result
    
def rungekutta4(func, init_x: dict, t_end: float, step: float, t_start=0, solver_args=(), func_args=()):
    ## input :
    # func(parameters, x_values) - function discribes the model
    # param - system parameters

    # init_x - initial system values
    # t_start - time of start, where initial is set
    # t_end - ending
    # step - grid step
    ## output :
    # result - system vaue on grid of time points
    steps_to_save = solver_args[0]
    T = t_end - t_start
    Nt = int(T / step)
    if steps_to_save is None:
        steps_to_save = list(range(1,Nt))
    if Nt<0:
        step = -step
        Nt = -Nt
        warnings.warn('RK4: Check the step direction and start/end times. Automaticaly reversed step direction.')
        #raise Error('Wrong step direction')
    val_temp = np.array(list(init_x.values()))
    
    def make_step(prev_val, func):
        val_temp=prev_val
        a1 = step * func(val_temp, t_start+j * step, *func_args)
        val_temp = prev_val + a1 * 0.5
        a2 = step * func(val_temp, t_start+(j + 0.5) * step, *func_args)
        val_temp = prev_val + a2 * 0.5
        a3 = step * func(val_temp, t_start+(j + 0.5) * step, *func_args)
        val_temp = prev_val + a3
        a4 = step * func(val_temp, t_start+(j + 1.0) * step, *func_args)
        result = prev_val + (a1 + 2 * a2 + 2 * a3 + a4) / 6
        return result #+ jumps_data[j]
    
    result_to_save = np.empty((len(steps_to_save), *(val_temp.shape)))
    current_state = copy.copy(val_temp)
    i=0
    for j in range(1, Nt + 1):
        current_state = copy.copy(make_step(current_state, func=func))
        if j == steps_to_save[i]:
             result_to_save[i] = copy.copy(current_state)
             i += 1
             try:
                   steps_to_save[i]
             except:
                   break
    return result_to_save


def integrate(func, step, t_start, t_end, **kwargs):
    T = t_end-t_start
    Nt = int(T/step)
    res = 0
    for i in range(Nt):
        res += step * func(t_start+step*i, **kwargs)
        print(res)
    return res

def eval_grads(derivatives_strings: dict, **kwargs,):
    res = [0]*len(derivatives_strings)
    for i, string in enumerate(derivatives_strings.values()):
        print(string)
        res[i] = integrate(lambda t, **kw: eval(string, kwargs | {'t':t}), **kwargs)
    return res


def precompile_model(
    equation_strings: dict,
    custom_vars: dict = {},
    ):
    
    _custom_vars = copy.copy(custom_vars)
    for key in _custom_vars.keys():
        try:
            _custom_vars[key] = compile(_custom_vars[key].replace(" ",""), "<string>", "eval",optimize=1) #eval(_custom_vars[key], aliases_dict)
        except TypeError:
            print("The problem with custom var: " + key + " : " + _custom_vars[key])
            raise TypeError
    for key in (equation_strings.keys()):
        try:
            equation_strings[key] = compile(equation_strings[key].replace(" ",""), "<string>", "eval",optimize=1)#eval(val, aliases_dict | _custom_vars)    
        except TypeError:
            raise TypeError("The problem with eq string: " + key + " : " + equation_strings[key])
    return equation_strings, _custom_vars


def ode_model_wrapped(x,t, *args):
    _1, equation_strings, _2 = args
    system_state = dict(zip(equation_strings.keys(), x))
    return ode_model(system_state, t, *args)

def ode_model(
    system_state: dict,
    t: float = 0,
    params: dict = {},
    equation_strings: dict = {},
    custom_vars: dict = {},
    multistart=None,
):
    ## input :
    # param - system parameters
    # sys_x - system values at indicated time point

    ## output :
    # result - system values at indicated time point

    _custom_vars = copy.copy(custom_vars)
    aliases_dict = params | system_state | {"t": t} | {"sc":sc}
    for key in _custom_vars.keys():
        try:
            _custom_vars[key] = eval(_custom_vars[key], aliases_dict)
        except TypeError:
            print("The problem with custom var: " + key + " : " + _custom_vars[key])
            raise TypeError
    
    #if multistart is not None:
    #    result = np.zeros((len(equation_strings), multistart))
    #else:
    #    result = np.zeros(len(equation_strings))
    result = None
    for i, val in enumerate(equation_strings.values()):
        try:
            v = eval(val, aliases_dict | _custom_vars)
            if result is None:
                if isinstance(v, float):
                    result = np.zeros(len(equation_strings))
                else:
                    result = np.zeros((len(equation_strings), *v.shape))
            else:
                result[i] = v
            if np.isnan(result[i]).any():
                pass
                #raise ValueError("The solution is nan: " + list(equation_strings.keys())[i] + " : " + val)
        except TypeError:
            raise TypeError("The problem with eq string: " + list(equation_strings.keys())[i] + " : " + val)
        #except ValueError:
        #    raise ValueError("The problem with eq string: " + list(equation_strings.keys())[i])
    return np.array(result)


def ode_solve(params, initial_state, t_end, step, equation_strings, custom_vars={}, t_start=0, solver=scipy_solver, steps_to_save=None):
    data = np.array(solver(func=ode_model_wrapped,
                init_x=initial_state,
                t_end=t_end,
                t_start=t_start,
                step=step,
                func_args=(params, equation_strings, custom_vars),
                solver_args=(steps_to_save,)
                )
    )
    points = np.array([
        np.linspace(t_start, t_end, num=data.shape[0], endpoint=True) for _ in range(data.shape[1])
    ])
    keys = np.array(list(initial_state.keys()))
    return Data(keys, points, data)


def loss_func(data_pred: Data, data_ex: Data, loss_custom_funcs=None):
    ## input :
    # data_pred - predicted data
    # data_ex - exact data

    ## output :
    # quadratic loss
    
    #try:
    #    data_pred = loss_custom_funcs(data_pred)
    #except:
    #    pass

    target_sum = 0
    inds = list(map(lambda x: x in data_ex.keys, data_pred.keys))

    data_pred.data = copy.deepcopy(data_pred.data[:,inds])
    data_pred.points = data_pred.points[inds]
    data_pred.keys = data_pred.keys[inds]

    if np.all(np.sort(data_pred.keys) != np.sort(data_ex.keys)):
        raise ValueError("Given exact data keys do not match model keys")

    for i in range(len(data_pred.keys)):
        p_ex = np.array(data_ex.points[i])
        p_pred = np.array(data_pred.points[i])
        step = np.min(np.abs(p_ex[1:] - p_ex[:-1]))/2
        step2 = np.min(np.abs(p_pred[1:] - p_pred[:-1]))/2
        s = min((step,step2))
        inds = list(map(lambda x: np.any(np.abs(x-data_ex.points[i])<s), data_pred.points[i]))
        inds2 = list(map(lambda x: np.any(np.abs(x-data_pred.points[i])<s), data_ex.points[i]))
        #print(np.sum(inds2))
        #print(np.sum(inds))
        dp = np.array(data_ex.data[i])[inds2]
        # dp_n = np.sqrt(np.sum(dp*dp))
        dp_n = np.sum(np.abs(dp) ** 2)
        target_sum += np.sum((data_pred.data[inds,i].T - dp).T ** 2) / dp_n
        
        #dp_n = np.sum(np.abs(dp))
        #target_sum += np.sum(np.abs(data_pred.data[inds,i].T - dp).T) / dp_n
    return target_sum
    # return np.sum((data_pred.data - data_ex.data)**2, (0,1))


def objective(data_ex: Data, model, default_params, trial_params, **kwargs):
    ## input :
    # data_ex - exact data

    ## output :
    # quadratic loss of model with indicated parameters and exact data

    params = copy.copy(default_params)
    params.update(trial_params)
    prediction = model(params=params, **kwargs)
    normed_pred = prediction  # prediction/np.asarray([np.max(prediction, prediction.shape[-1]).T]).T
    normed_ex = data_ex  # data_ex/np.asarray([np.max(data_ex, data_ex.shape[-1]).T]).T
    return loss_func(normed_pred, normed_ex)


def ode_objective(**kwargs):
    return objective(model=ode_solve, **kwargs)
