# tb-bayes
Mathematical model of Tuberculosis transmission and probability forecasting

## Usage

Model description is contained in description file `tb-mbr\parameters.yaml`.

The direct problem is solved with `default-parameters` and `initial_state` from description file. 

The inverse problem is solved with parameters restored from areas defined in `estim_and_bounds`. Other paramters are picked from `default-parameters`.
The `passed_keys` are the names of the compartmets with datapoints, used in solution of inverse problem.

`db_keys` are used to pick data from database.

To run direct problem solution run

```
python direct_solve.py
```


To run inverse problem solution run

```
python inverse_solve.py
```

(database required)