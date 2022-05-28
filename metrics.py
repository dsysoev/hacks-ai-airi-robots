import pandas as pd
from joblib import Parallel, delayed

from main import main


for map_size, iteration in [('tiny', 300), ('huge', 42), ('hard', 14)]:
    results = Parallel(n_jobs=7)(delayed(main)(anim=False, seed=i, map_size=map_size) for i in range(iteration))
    results = pd.DataFrame(results)
    results.describe().round(2).to_markdown('metrics/{}_overall.md'.format(map_size))

    results.sort_values('steps', ascending=False).head(10).to_markdown('metrics/{}_top_worst.md'.format(map_size))
