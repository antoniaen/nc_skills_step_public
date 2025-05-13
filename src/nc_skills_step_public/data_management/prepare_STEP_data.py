"""Function(s) for cleaning the data set(s)."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def rename_variables(data):
    """Rename variables.

    Args:
        data (pandas DataFrame): The data.


    Returns:
        changed_data (pandas DataFrame): The changed data.

    """
    changed_data = data.rename(
        columns={
            "gender": "female",
            "m1a_5a1": "brth_day",
            "m1a_q05m": "brth_month",
            "m1a_q05y": "brth_year",
            "age_start": "age_strt_school",
            "m2_q29": "age_end_educ",
            "conscientiousness_avg": "conscientiousness_av",
            "ses": "ses_age15",
            "m7a_q23": "fam_econ_1_to_10_age15",
            "m7a_q25": "worked_age15",
            "shocks": "shocks_age15",
            "shocks_dummy": "shocks_dummy_age15",
            "old_brothers": "old_brothers_age12",
            "old_sisters": "old_sisters_age12",
            "young_brothers": "young_brothers_age12",
            "young_sisters": "young_sisters_age12",
            "m7a_q27": "abuse_outsidehh_age15",
            "m7a_q28": "abuse_insidehh_age15",
        },
    )

    return changed_data


def clean_data(data):
    """Clean data. Handle missing data etc.

    Args:
        data (pandas DataFrame): The data.

    Returns:
        changed_data (pandas DataFrame): The changed data.

    """
    changed_data = data.copy()

    changed_data[["brth_month", "age_strt_school"]] = changed_data[
        ["brth_month", "age_strt_school"]
    ].replace({88: np.nan, 99: np.nan, 97: np.nan, -66: np.nan})

    changed_data["age_end_educ"] = changed_data["age_end_educ"].replace(
        {92: np.nan, 97: np.nan},
    )

    changed_data["brth_year"] = changed_data["brth_year"].replace(8888, np.nan)

    changed_data["parental"] = changed_data["parental"].replace(
        {
            1: "Yes, always or almost always",
            2: "Yes, sometimes",
            3: "No, never or almost never",
            4: "No, never or almost never",
            8: np.nan,
            9: np.nan,
            -1: np.nan,
        },
    )

    changed_data["fam_econ_1_to_10_age15"] = changed_data[
        "fam_econ_1_to_10_age15"
    ].replace({-3: np.nan, -6: np.nan, -9: np.nan, 0: np.nan, 97: np.nan})

    changed_data["worked_age15"] = changed_data["worked_age15"].replace(
        {1: 1, 2: 0, 0: np.nan, -6: np.nan, 7: np.nan},
    )

    changed_data["occupation"] = changed_data["occupation"].replace(
        {
            1: "1 Managers",
            2: "2 Professionals",
            3: "3 Technicians and associate professionals",
            4: "4 Clerical support workers",
            5: "5 Service and sales workers",
            6: "6 Skilled agricultural, forestry and fishery workers",
            7: "7 Craft and related trades workers",
            8: "8 Plant and machine operators, and assemblers",
            9: "9 Elementary occupations",
            0: "0 Armed forces occupations",
        },
    )
    occupation_cat = [
        "1 Managers",
        "2 Professionals",
        "3 Technicians and associate professionals",
        "4 Clerical support workers",
        "5 Service and sales workers",
        "6 Skilled agricultural, forestry and fishery workers",
        "7 Craft and related trades workers",
        "8 Plant and machine operators, and assemblers",
        "9 Elementary occupations",
        "0 Armed forces occupations",
    ]
    changed_data["occupation"] = pd.Categorical(
        changed_data["occupation"],
        categories=occupation_cat,
        ordered=True,
    )

    changed_data["occtype_step"] = changed_data["occtype_step"].replace(
        {
            1: "Highly skilled white collar - Managers/Professionals/Technicians",
            2: "Low skilled white collar",
            3: "Crafts and related trades workers; Plant and machine operator and assemblers",
            4: "Elementary occupations",
            5: "Skilled agriculture work",
            0: "Military personnel",
        },
    )
    occtype_step_cat = [
        "Highly skilled white collar - Managers/Professionals/Technicians",
        "Low skilled white collar",
        "Crafts and related trades workers; Plant and machine operator and assemblers",
        "Elementary occupations",
        "Skilled agriculture work",
        "Military personnel",
    ]
    changed_data["occtype_step"] = pd.Categorical(
        changed_data["occtype_step"],
        categories=occtype_step_cat,
        ordered=True,
    )

    changed_data["abuse_outsidehh_age15"] = changed_data[
        "abuse_outsidehh_age15"
    ].replace({1: 1, 2: 0, 9: np.nan})
    changed_data["abuse_insidehh_age15"] = changed_data["abuse_insidehh_age15"].replace(
        {1: 1, 2: 0, 9: np.nan},
    )

    return changed_data


def add_data_columns(data):
    """Add variables.

    Args:
        data (pandas DataFrame): The data.

    Returns:
        more_data (pandas DataFrame): The data with added columns.

    """
    more_data = data.copy()

    more_data["brth_year2"] = more_data["brth_year"] ** 2

    more_data["patience_binary"] = np.where(
        more_data["m6b_q04"] == 2,
        1,
        np.where(more_data["m6b_q04"] == 1, 0, np.nan),
    )

    more_data["risk_binary"] = np.where(
        more_data["m6b_q01"] == 2,
        1,
        np.where(more_data["m6b_q01"] == 1, 0, np.nan),
    )

    more_data["parents_info_school"] = np.where(
        (data["parental"] == "Yes, always or almost always")
        | (data["parental"] == "Yes, sometimes"),
        1,
        np.where(more_data["parental"] == "No, never or almost never", 0, np.nan),
    )

    more_data["years_educ_calc"] = (
        more_data["age_end_educ"] - more_data["age_strt_school"]
    )
    more_data["years_educ_calc"] = np.where(
        more_data["years_educ_calc"] < 0,
        np.nan,
        more_data["years_educ_calc"],
    )
    more_data["years_educ_calc"] = np.where(
        more_data["years_educ_calc"] > 30,
        31,
        more_data["years_educ_calc"],
    )

    more_data["siblings_age12"] = (
        more_data["old_brothers_age12"]
        + more_data["old_sisters_age12"]
        + more_data["young_brothers_age12"]
        + more_data["young_sisters_age12"]
    )

    more_data["young_siblings_age12"] = (
        more_data["young_brothers_age12"] + more_data["young_sisters_age12"]
    )

    more_data["old_siblings_age12"] = (
        more_data["old_brothers_age12"] + more_data["old_sisters_age12"]
    )

    more_data["overweight"] = np.where(
        (more_data["BMI_class"] == 3) | (more_data["BMI_class"] == 4),
        1,
        np.where(
            (more_data["BMI_class"] == 1) | (more_data["BMI_class"] == 2),
            0,
            np.nan,
        ),
    )

    more_data["abuse_any_age15"] = np.where(
        (more_data["abuse_outsidehh_age15"] == 1)
        | (more_data["abuse_insidehh_age15"] == 1),
        1,
        np.where(
            (more_data["abuse_outsidehh_age15"] == 0)
            & (more_data["abuse_insidehh_age15"] == 0),
            0,
            np.nan,
        ),
    )

    return more_data


def harmonize_skill_items(data):
    """Harmonize (recode) skill items. Wave 1 and wave 2 items are reversed.

    I harmonize items across survey waves and at the same time across underlying traits
    (for personality traits). The latter means that answers to reversely asked questions
    are recoded s.t. a larger number corresponds to a higher level of the underlying trait.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_r (pandas DataFrame): The data with harmonized items instead.

    """
    wave1_to_recode = _skill_items_dicts_and_lists(which="wave1_to_recode")

    wave2_to_recode = _skill_items_dicts_and_lists(which="wave2_to_recode")

    data_r = data.copy()

    for item in wave1_to_recode:
        data_r[wave1_to_recode[item]] = data_r[item]
    for item in wave2_to_recode:
        data_r[wave2_to_recode[item]] = data_r[item]

    for item in wave1_to_recode:
        data_r.loc[
            data_r["country"].isin(
                [
                    "Bolivia",
                    "Colombia",
                    "Laos",
                    "Sri Lanka",
                    "Vietnam",
                    "Yunnan",
                    "Ukraine",
                ],
            ),
            wave1_to_recode[item],
        ] = data_r.loc[
            data_r["country"].isin(
                [
                    "Bolivia",
                    "Colombia",
                    "Laos",
                    "Sri Lanka",
                    "Vietnam",
                    "Yunnan",
                    "Ukraine",
                ],
            ),
            item,
        ].replace(
            {
                1: 4,
                2: 3,
                3: 2,
                4: 1,
            },
        )

    for item in wave2_to_recode:
        data_r.loc[
            data_r["country"].isin(
                ["Armenia", "Georgia", "Ghana", "Macedonia", "Kenya"],
            ),
            wave2_to_recode[item],
        ] = data_r.loc[
            data_r["country"].isin(
                ["Armenia", "Georgia", "Ghana", "Macedonia", "Kenya"],
            ),
            item,
        ].replace(
            {
                1: 4,
                2: 3,
                3: 2,
                4: 1,
            },
        )

    data_r = data_r.drop(
        columns=list(wave1_to_recode.keys()) + list(wave2_to_recode.keys()),
    )

    return data_r


def standardize_skills_and_prefs(data):
    """Standardize skills and preferences.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_s (pandas DataFrame): The data with standardized variables in addition.

    """
    to_standardize = (
        # All skill averages
        [col for col in data if col.endswith("_av")]
        + [
            "risk",
            "discount",
            "write",
            "read",
            "num",
            "patience_binary",
            "risk_binary",
        ]
        # All skill items
        + [col for col in data if col.endswith("_h")]
        # All plausible values from the literacy test scores.
        + [col for col in data if col.startswith("PVLIT")]
    )
    data_s = data.copy()

    for col in to_standardize:
        data_s[col + "_s"] = _standardize(col=data_s[col])

    return data_s


def get_some_skills_with_pca(data):
    """Use PCA to get personality traits and behaviors.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_pca (pandas DataFrame): The data with principal components in addition.

    """
    orig_data = data.copy()

    skill_dict = _skill_items_dicts_and_lists(which="skill_dict")

    for skill in skill_dict:
        # Prepare data.
        data_pca = orig_data[skill_dict[skill]].copy()
        data_pca = data_pca.dropna()
        # Run PCA.
        my_pca = PCA(n_components=1)
        my_pca_component = my_pca.fit_transform(data_pca)
        # Flip sign if all loadings are negative.
        if np.all(my_pca.components_ < 0):
            my_pca_component = -my_pca_component
        # Add resulting component to the original data.
        orig_data[skill + "_pca"] = np.nan
        orig_data.loc[data_pca.index, skill + "_pca"] = my_pca_component

    return orig_data


def get_acquiescence_bias_corrected_skills(data):
    """Correct for acquiescence bias in skills.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_acq (pandas DataFrame): The data with corrected skills in addition.

    """
    data_acq = data.copy()

    # Get the individual acquiescence biases (AB).
    data_acq["acq_bias"] = _estimate_acquiescence_bias(
        data=data_acq,
        reversed_info=_skill_items_dicts_and_lists(which="reversed_info"),
    )

    # Correct each item for AB.
    for item in _skill_items_dicts_and_lists(which="reversed_item_list"):
        data_acq[item + "_acq_corr"] = data_acq[item] + data_acq["acq_bias"]
    for item in _skill_items_dicts_and_lists(which="non_reversed_item_list"):
        data_acq[item + "_acq_corr"] = data_acq[item] - data_acq["acq_bias"]

    # Calculate the individual skill measure by taking the average of the corrected items.
    skill_dict = _skill_items_dicts_and_lists(which="skill_dict")
    acq_skill_dict = {
        key: [item.replace("_s", "_acq_corr") for item in value]
        for key, value in skill_dict.items()
    }
    for skill in acq_skill_dict:
        data_acq[skill + "_av_acq_corr"] = data_acq[acq_skill_dict[skill]].mean(axis=1)

    # Standardize the corrected skill measures.
    for skill in acq_skill_dict:
        data_acq[skill + "_av_s_abcorr"] = _standardize(
            data_acq[skill + "_av_acq_corr"],
        )

    data_acq = data_acq.drop(
        columns=[col for col in data_acq if col.endswith("_av_acq_corr")],
    )

    return data_acq


def get_skills_based_on_laajaj_et_al(data):
    """Get skills based on Laajaj et al. (2019).

    In one case we only use items loading on the correct skill, in the other we use
    all items but for the skill they are loading on not for the skill they are designed for.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_laajaj (pandas DataFrame): The data with Laajaj et al. skills in addition.

    """
    data_laajaj = data.copy()

    laajaj_et_al_drop = _skill_items_dicts_and_lists(which="laajaj_et_al_drop")
    laajaj_et_al_replace = _skill_items_dicts_and_lists(which="laajaj_et_al_replace")

    for skill in laajaj_et_al_drop["Ghana"]:
        data_laajaj[skill + "_av_l_drop"] = np.where(
            data_laajaj["country"] == "Ghana",
            data_laajaj[laajaj_et_al_drop["Ghana"][skill]].mean(axis=1),
            np.where(
                data_laajaj["country"] == "Vietnam",
                data_laajaj[laajaj_et_al_drop["Vietnam"][skill]].mean(axis=1),
                np.where(
                    data_laajaj["country"] == "Bolivia",
                    data_laajaj[laajaj_et_al_drop["Bolivia"][skill]].mean(axis=1),
                    np.where(
                        data_laajaj["country"] == "Colombia",
                        data_laajaj[laajaj_et_al_drop["Colombia"][skill]].mean(axis=1),
                        np.nan,
                    ),
                ),
            ),
        )
    for skill in laajaj_et_al_replace["Ghana"]:
        data_laajaj[skill + "_av_l_replace"] = np.where(
            data_laajaj["country"] == "Ghana",
            data_laajaj[laajaj_et_al_replace["Ghana"][skill]].mean(axis=1),
            np.where(
                data_laajaj["country"] == "Vietnam",
                data_laajaj[laajaj_et_al_replace["Vietnam"][skill]].mean(axis=1),
                np.where(
                    data_laajaj["country"] == "Bolivia",
                    data_laajaj[laajaj_et_al_replace["Bolivia"][skill]].mean(axis=1),
                    np.where(
                        data_laajaj["country"] == "Colombia",
                        data_laajaj[laajaj_et_al_replace["Colombia"][skill]].mean(
                            axis=1,
                        ),
                        np.nan,
                    ),
                ),
            ),
        )

    # Standardize the adjusted skill measures.
    for skill in laajaj_et_al_drop["Ghana"]:
        data_laajaj[skill + "_av_s_laajaj_drop"] = _standardize(
            data_laajaj[skill + "_av_l_drop"],
        )
    for skill in laajaj_et_al_replace["Ghana"]:
        data_laajaj[skill + "_av_s_laajaj_replace"] = _standardize(
            data_laajaj[skill + "_av_l_replace"],
        )

    data_laajaj = data_laajaj.drop(
        columns=[
            col for col in data_laajaj if col.endswith(("_av_l_drop", "_av_l_replace"))
        ],
    )

    return data_laajaj


def create_skill_weights(data):
    """Create weights for the skill measures.

    Measures based on more items are given more weight.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_w (pandas DataFrame): The data with skill weights in addition.

    """
    data_w = data.copy()

    skill_dict = _skill_items_dicts_and_lists(which="skill_dict")

    for skill in skill_dict:
        data_w[skill + "_num_items"] = data_w[skill_dict[skill]].notna().sum(axis=1)

        data_w[skill + "_weight"] = np.where(
            data_w[skill + "_num_items"] == 1,
            1,
            np.where(
                data_w[skill + "_num_items"] == 2,
                1.12,
                np.where(
                    data_w[skill + "_num_items"] == 3,
                    1.19,
                    np.where(data_w[skill + "_num_items"] == 4, 1.24, 0),
                ),
            ),
        )

    data_w = data_w.drop(columns=[col for col in data_w if col.endswith("_num_items")])

    return data_w


def _standardize(col):
    """Standardize a specified column of a dataframe.

    Args:
        col (series): Column of a pandas dataframe.

    Returns:
        column (series): Standardised column.

    """
    column = (col - col.mean()) / col.std()
    return column


def _estimate_acquiescence_bias(data, reversed_info):
    """Estimate acquiescence bias in the data set.

    Args:
        data (pandas DataFrame): The data set.
        reversed_info (dict): Indicates which items are reversed and which are not.

    Returns:
        acq_bias (pandas Series): Acquiescence bias.

    """
    av_reversed = {"extraversion": None, "conscientiousness": None, "stability": None}
    av_non_reversed = {
        "extraversion": None,
        "conscientiousness": None,
        "stability": None,
    }
    differences = {"extraversion": None, "conscientiousness": None, "stability": None}

    for trait in reversed_info:
        av_reversed[trait] = data[reversed_info[trait]["reversed"]].mean(axis=1)
        av_non_reversed[trait] = data[reversed_info[trait]["non_reversed"]].mean(axis=1)
        differences[trait] = (av_non_reversed[trait] - av_reversed[trait]) / 2

    acq_bias = (
        differences["extraversion"]
        + differences["conscientiousness"]
        + differences["stability"]
    ) / 3

    return acq_bias


def _skill_items_dicts_and_lists(which):
    """Get dictionaries and lists of skill items.

    Args:
        which (str): Which items to get.

    Returns:
        (dict or list): The requested items.

    """
    wave1_to_recode = {
        "m6a_q0101": "e1_h",
        "m6a_q0120": "e3_h",
        "m6a_q0102": "c1_h",
        "m6a_q0117": "c3_h",
        "m6a_q0103": "o1_h",
        "m6a_q0111": "o2_h",
        "m6a_q0114": "o3_h",
        "m6a_q0105": "s1_h",
        "m6a_q0109": "a1_h",
        "m6a_q0116": "a2_h",
        "m6a_q0119": "a3_h",
        "m6a_q0106": "g1_h",
        "m6a_q0108": "g2_h",
        "m6a_q0113": "g3_h",
        "m6a_q0115": "d1_h",
        "m6a_q0121": "d2_h",
        "m6a_q0123": "d3_h",
        "m6a_q0124": "d4_h",
        "m6a_q0107": "h1_h",
        "m6a_q0122": "h2_h",
    }
    wave2_to_recode = {
        "m6a_q0104": "e2_h",
        "m6a_q0112": "c2_h",
        "m6a_q0110": "s2_h",
        "m6a_q0118": "s3_h",
    }
    skill_dict = {
        "extraversion": ["e1_h_s", "e2_h_s", "e3_h_s"],
        "conscientiousness": ["c1_h_s", "c2_h_s", "c3_h_s"],
        "openness": ["o1_h_s", "o2_h_s", "o3_h_s"],
        "stability": ["s1_h_s", "s2_h_s", "s3_h_s"],
        "agreeableness": ["a1_h_s", "a2_h_s", "a3_h_s"],
        "grit": ["g1_h_s", "g2_h_s", "g3_h_s"],
        "decision": ["d1_h_s", "d2_h_s", "d3_h_s", "d4_h_s"],
        "hostile": ["h1_h_s", "h2_h_s"],
    }
    reversed_item_list = list(wave2_to_recode.values())
    non_reversed_item_list = list(wave1_to_recode.values())
    reversed_info = {
        "extraversion": {"non_reversed": ["e1_h", "e3_h"], "reversed": ["e2_h"]},
        "conscientiousness": {"non_reversed": ["c1_h", "c3_h"], "reversed": ["c2_h"]},
        "stability": {"non_reversed": ["s1_h"], "reversed": ["s2_h", "s3_h"]},
    }
    laajaj_et_al_drop = {
        "Ghana": {
            "openness": ["o2_h_acq_corr", "o3_h_acq_corr"],
            "conscientiousness": ["c2_h_acq_corr"],
            "extraversion": ["e1_h_acq_corr", "e2_h_acq_corr", "e3_h_acq_corr"],
            "agreeableness": [],
            "stability": ["s1_h_acq_corr", "s2_h_acq_corr", "s3_h_acq_corr"],
        },
        "Vietnam": {
            "openness": ["o1_h_acq_corr", "o2_h_acq_corr", "o3_h_acq_corr"],
            "conscientiousness": ["c1_h_acq_corr", "c2_h_acq_corr", "c3_h_acq_corr"],
            "extraversion": ["e2_h_acq_corr", "e3_h_acq_corr"],
            "agreeableness": ["a3_h_acq_corr"],
            "stability": ["s1_h_acq_corr", "s2_h_acq_corr", "s3_h_acq_corr"],
        },
        "Bolivia": {
            "openness": ["o1_h_acq_corr", "o2_h_acq_corr", "o3_h_acq_corr"],
            "conscientiousness": ["c1_h_acq_corr", "c2_h_acq_corr", "c3_h_acq_corr"],
            "extraversion": ["e1_h_acq_corr", "e2_h_acq_corr", "e3_h_acq_corr"],
            "agreeableness": ["a1_h_acq_corr", "a2_h_acq_corr", "a3_h_acq_corr"],
            "stability": ["s1_h_acq_corr", "s2_h_acq_corr", "s3_h_acq_corr"],
        },
        "Colombia": {
            "openness": [],
            "conscientiousness": ["c1_h_acq_corr", "c3_h_acq_corr"],
            "extraversion": ["e1_h_acq_corr", "e2_h_acq_corr", "e3_h_acq_corr"],
            "agreeableness": ["a1_h_acq_corr", "a2_h_acq_corr", "a3_h_acq_corr"],
            "stability": ["s1_h_acq_corr", "s2_h_acq_corr", "s3_h_acq_corr"],
        },
    }
    laajaj_et_al_replace = {
        "Ghana": {
            "openness": [
                "o2_h_acq_corr",
                "o3_h_acq_corr",
                "c1_h_acq_corr",
                "a1_h_acq_corr",
                "a2_h_acq_corr",
                "a3_h_acq_corr",
            ],
            "conscientiousness": ["c2_h_acq_corr"],
            "extraversion": ["e1_h_acq_corr", "e2_h_acq_corr", "e3_h_acq_corr"],
            "agreeableness": ["o1_h_acq_corr", "c3_h_acq_corr"],
            "stability": ["s1_h_acq_corr", "s2_h_acq_corr", "s3_h_acq_corr"],
        },
        "Vietnam": {
            "openness": [
                "o1_h_acq_corr",
                "o2_h_acq_corr",
                "o3_h_acq_corr",
                "a2_h_acq_corr",
            ],
            "conscientiousness": ["c1_h_acq_corr", "c2_h_acq_corr", "c3_h_acq_corr"],
            "extraversion": ["e2_h_acq_corr", "e3_h_acq_corr", "a1_h_acq_corr"],
            "agreeableness": ["a3_h_acq_corr", "e1_h_acq_corr"],
            "stability": ["s1_h_acq_corr", "s2_h_acq_corr", "s3_h_acq_corr"],
        },
        "Bolivia": {
            "openness": ["o1_h_acq_corr", "o2_h_acq_corr", "o3_h_acq_corr"],
            "conscientiousness": ["c1_h_acq_corr", "c2_h_acq_corr", "c3_h_acq_corr"],
            "extraversion": ["e1_h_acq_corr", "e2_h_acq_corr", "e3_h_acq_corr"],
            "agreeableness": ["a1_h_acq_corr", "a2_h_acq_corr", "a3_h_acq_corr"],
            "stability": ["s1_h_acq_corr", "s2_h_acq_corr", "s3_h_acq_corr"],
        },
        "Colombia": {
            "openness": ["c2_h_acq_corr"],
            "conscientiousness": ["c1_h_acq_corr", "c3_h_acq_corr", "o2_h_acq_corr"],
            "extraversion": [
                "e1_h_acq_corr",
                "e2_h_acq_corr",
                "e3_h_acq_corr",
                "o1_h_acq_corr",
            ],
            "agreeableness": [
                "a1_h_acq_corr",
                "a2_h_acq_corr",
                "a3_h_acq_corr",
                "o3_h_acq_corr",
            ],
            "stability": ["s1_h_acq_corr", "s2_h_acq_corr", "s3_h_acq_corr"],
        },
    }

    return locals()[which]
