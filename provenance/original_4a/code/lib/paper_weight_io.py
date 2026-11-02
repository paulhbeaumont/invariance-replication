"""Route paper outputs to the explicitly selected, hash-pinned weighting baseline."""
from baseline_io import APP,DATA,read_json,require,sha
path=APP/'config/paper_weights.json'
POPULATION=path.exists()
if POPULATION:
    spec=read_json(path)
    require(spec['id']=='population1975' and spec['status']=='approved','Unknown paper weighting baseline')
    require(spec['population_sha256']==sha(DATA/'analysis/county_population_1975.parquet'),'Paper population weights changed')
COEFFICIENTS=APP/'results/coefficients'/('population1975' if POPULATION else '')
MANIFESTS=APP/'results/manifests'/('population1975' if POPULATION else '')
STYLE=APP/'results/tables'/('log_disaggregation_population1975' if POPULATION else 'log_disaggregation_review')
