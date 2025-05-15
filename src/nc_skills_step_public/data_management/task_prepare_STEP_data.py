"""Merge the different countries into one dataset.

Prepare this dataset.

"""

import pandas as pd
import pytask

from nc_skills_step_public.config import BLD
from nc_skills_step_public.data_management import prepare_STEP_data as prd


@pytask.mark.depends_on(
    {
        "scripts": ["prepare_STEP_data.py"],
        "Armenia": BLD / "python" / "data" / "Armenia_small.pkl",
        "Bolivia": BLD / "python" / "data" / "Bolivia_small.pkl",
        "Colombia": BLD / "python" / "data" / "Colombia_small.pkl",
        "Georgia": BLD / "python" / "data" / "Georgia_small.pkl",
        "Ghana": BLD / "python" / "data" / "Ghana_small.pkl",
        "Kenya": BLD / "python" / "data" / "Kenya_small.pkl",
        "Laos": BLD / "python" / "data" / "Laos_small.pkl",
        "Macedonia": BLD / "python" / "data" / "Macedonia_small.pkl",
        "Sri_Lanka": BLD / "python" / "data" / "Sri_Lanka_small.pkl",
        "Ukraine": BLD / "python" / "data" / "Ukraine_small.pkl",
        "Vietnam": BLD / "python" / "data" / "Vietnam_small.pkl",
        "Yunnan": BLD / "python" / "data" / "Yunnan_small.pkl",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "STEP_data_clean.pkl")
def task_merge_and_prepare_countries(depends_on, produces):
    """Merge, clean and prepare the data."""
    datasets = [
        pd.read_pickle(depends_on[country]) for country in list(depends_on.keys())[1:13]
    ]
    data_appended = pd.concat(datasets, ignore_index=True)

    data_renamed = prd.rename_variables(data_appended)
    data_cleaned = prd.clean_data(data_renamed)
    data_new_columns = prd.add_data_columns(data_cleaned)
    data_har_items = prd.harmonize_skill_items(data_new_columns)
    data_w_std = prd.standardize_skills_and_prefs(data_har_items)
    data_w_pca = prd.get_some_skills_with_pca(data_w_std)
    data_ab_corr = prd.get_acquiescence_bias_corrected_skills(data_w_pca)
    data_laajaj = prd.get_skills_based_on_laajaj_et_al(data_ab_corr)
    data_weights = prd.create_skill_weights(data_laajaj)

    data_weights.to_pickle(produces)
