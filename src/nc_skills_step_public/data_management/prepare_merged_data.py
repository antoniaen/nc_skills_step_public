import numpy as np

from nc_skills_step_public import global_info as gl


def create_treatment_indicator(data, var_name, placebo=0):
    """Create a variable indicating treatment of compulsory schooling reform.

    The time window is 10 years before and after the pivotal cohort.

    Args:
        data (pandas DataFrame): The data containing STEP data and reforms.
        var_name (str): The name of the variable to be created.
        placebo (int): The distance to the true pivotal cohort.

    Returns:
        data_copy (pandas DataFrame): The adjusted data with treatment indicator.

    """
    data_copy = data.copy()

    data_copy[var_name] = np.where(
        (
            _treatment_reform_condition(
                data=data_copy,
                years=10,
                n_reform=1,
                placebo=placebo,
            )
        )
        | (
            _treatment_reform_condition(
                data=data_copy,
                years=10,
                n_reform=2,
                placebo=placebo,
            )
        ),
        1,
        np.where(
            (
                _control_reform_condition(
                    data=data_copy,
                    years=10,
                    n_reform=1,
                    placebo=placebo,
                )
            )
            | (
                _control_reform_condition(
                    data=data_copy,
                    years=10,
                    n_reform=2,
                    placebo=placebo,
                )
            ),
            0,
            np.nan,
        ),
    )

    return data_copy


def create_treatment_indicator_w_month(data):
    """Create a variable indicating treatment of compulsory schooling reform.

    Birth month is taken into account. The time window is 10 years before and after the
    pivotal cohort.

    Args:
        data (pandas DataFrame): The data containing STEP data and reforms.

    Returns:
        data_copy (pandas DataFrame): The adjusted data with treatment indicator.

    """
    data_copy = data.copy()

    data_copy["treated_w_month"] = np.where(
        (_treatment_reform_condition_w_month(data=data_copy, years=10, n_reform=1))
        | (_treatment_reform_condition_w_month(data=data_copy, years=10, n_reform=2)),
        1,
        np.where(
            (_control_reform_condition_w_month(data=data_copy, years=10, n_reform=1))
            | (_control_reform_condition_w_month(data=data_copy, years=10, n_reform=2)),
            0,
            np.nan,
        ),
    )

    return data_copy


def create_individuals_relevant_reform(data, var_name, placebo=0):
    """Create a variable indicating which reform is relevant for the individual.

    A reform is relevant if the individual belongs to either treatment or control
    group due to this reform. Note: the definition is closely linked to the
    create_treatment_indicator function.

    Args:
        data (pandas DataFrame): The data containing STEP data and reforms.
        var_name (str): The name of the variable to be created.
        placebo (int): The distance to the true pivotal cohort.

    Returns:
        data_copy (pandas DataFrame): The adjusted data with reform indicator.

    """
    data_copy = data.copy()

    data_copy[var_name] = np.where(
        # Belongs to treatment/control group because of reform 1.
        (
            _treatment_reform_condition(
                data=data_copy,
                years=10,
                n_reform=1,
                placebo=placebo,
            )
        )
        | (
            _control_reform_condition(
                data=data_copy,
                years=10,
                n_reform=1,
                placebo=placebo,
            )
        ),
        data_copy["country"].astype(str)
        + data_copy["reform_year_reform1"].fillna(0).astype(int).astype(str),
        # Belongs to treatment/control group because of reform 2.
        np.where(
            (
                _treatment_reform_condition(
                    data=data_copy,
                    years=10,
                    n_reform=2,
                    placebo=placebo,
                )
            )
            | (
                _control_reform_condition(
                    data=data_copy,
                    years=10,
                    n_reform=2,
                    placebo=placebo,
                )
            ),
            data_copy["country"].astype(str)
            + data_copy["reform_year_reform2"].fillna(0).astype(int).astype(str),
            np.nan,
        ),
    )

    # Cluster level variable.
    data_copy[var_name + "_brth_year"] = (
        data_copy[var_name] + "_" + data_copy["brth_year"].astype(str)
    )

    return data_copy


def create_individuals_relevant_reform_months_based(data, var_name):
    """Create a variable indicating which reform is relevant for the individual.

    A reform is relevant if the individual belongs to either treatment or control
    group due to this reform. Note: the definition is closely linked to the
    create_treatment_indicator function.

    Args:
        data (pandas DataFrame): The data containing STEP data and reforms.
        var_name (str): The name of the variable to be created.

    Returns:
        data_copy (pandas DataFrame): The adjusted data with reform indicator.

    """
    data_copy = data.copy()

    data_copy[var_name] = np.where(
        # Belongs to treatment/control group because of reform 1.
        (
            _treatment_reform_condition_w_month(
                data=data_copy,
                years=10,
                n_reform=1,
            )
        )
        | (
            _control_reform_condition_w_month(
                data=data_copy,
                years=10,
                n_reform=1,
            )
        ),
        data_copy["country"].astype(str)
        + data_copy["reform_year_reform1"].fillna(0).astype(int).astype(str),
        # Belongs to treatment/control group because of reform 2.
        np.where(
            (
                _treatment_reform_condition_w_month(
                    data=data_copy,
                    years=10,
                    n_reform=2,
                )
            )
            | (
                _control_reform_condition_w_month(
                    data=data_copy,
                    years=10,
                    n_reform=2,
                )
            ),
            data_copy["country"].astype(str)
            + data_copy["reform_year_reform2"].fillna(0).astype(int).astype(str),
            np.nan,
        ),
    )

    # Cluster level variable.
    data_copy[var_name + "_brth_year"] = (
        data_copy[var_name] + "_" + data_copy["brth_year"].astype(str)
    )

    return data_copy


def create_partially_treated_indicator(data):
    """Create a variable indicating partially treated individuals.

    These are individuals who are actually born after the cutoff but which might not
    yet be affected due to loose enforcement of the reform.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_copy (pandas DataFrame): The data with the additional column.

    """
    data_copy = data.copy()
    data_copy["partially_treated"] = np.where(
        (data_copy["country_reform"] == "Vietnam1991")
        & (data_copy["brth_year"].isin(range(1977, 1981))),
        1,
        0,
    )

    return data_copy


def create_partially_treated_placebo_indicator(data, placebo_number):
    """Create a variable indicating placebo 'partially treated' individuals.

    Args:
        data (pandas DataFrame): The data set.
        placebo_number (string): The placebo number as string.

    Returns:
        data_copy (pandas DataFrame): The data with the additional column.

    """
    data_copy = data.copy()
    data_copy["partially_treated_placebo" + placebo_number] = np.where(
        (data_copy["country_reform_placebo" + placebo_number] == "Vietnam1991")
        & (data_copy["rel_placebo_cohort" + placebo_number].isin(range(4))),
        1,
        0,
    )

    return data_copy


def create_relative_cohort(data):
    """Create a variable indicating the distance to the pivotal cohort.

    This variable is for the 10-years-window treatment indicator. In addition, a
    squared term is created.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_copy (pandas DataFrame): The data with the additional columns.

    """
    data_copy = data.copy()

    data_copy["rel_cohort"] = np.where(
        (_treatment_reform_condition(data=data_copy, years=10, n_reform=1))
        | (_control_reform_condition(data=data_copy, years=10, n_reform=1)),
        data_copy["brth_year"] - data_copy["pivotal_lower_reform1"],
        np.where(
            (_treatment_reform_condition(data=data_copy, years=10, n_reform=2))
            | (_control_reform_condition(data=data_copy, years=10, n_reform=2)),
            data_copy["brth_year"] - data_copy["pivotal_lower_reform2"],
            np.nan,
        ),
    )

    data_copy["rel_cohort2"] = data_copy["rel_cohort"] ** 2
    data_copy["rel_cohort3"] = data_copy["rel_cohort"] ** 3
    data_copy["rel_cohort4"] = data_copy["rel_cohort"] ** 4

    return data_copy


def create_relative_placebo_cohort(data):
    """Create a variable indicating the distance to the pivotal placebo cohort.

    This variable is for a 5-years-window treatment indicator. In addition, a
    squared term is created.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_copy (pandas DataFrame): The data with the additional columns.

    """
    data_copy = data.copy()

    for year in gl.placebo_years:
        data_copy["rel_placebo_cohort" + gl.placebo_years[year]] = np.where(
            (
                _treatment_reform_condition(
                    data=data_copy,
                    years=5,
                    n_reform=1,
                    placebo=year,
                )
            )
            | (
                _control_reform_condition(
                    data=data_copy,
                    years=5,
                    n_reform=1,
                    placebo=year,
                )
            ),
            data_copy["brth_year"] - (data_copy["pivotal_lower_reform1"] + year),
            np.where(
                (
                    _treatment_reform_condition(
                        data=data_copy,
                        years=5,
                        n_reform=2,
                        placebo=year,
                    )
                )
                | (
                    _control_reform_condition(
                        data=data_copy,
                        years=5,
                        n_reform=2,
                        placebo=year,
                    )
                ),
                data_copy["brth_year"] - (data_copy["pivotal_lower_reform2"] + year),
                np.nan,
            ),
        )

        data_copy["rel_placebo_cohort" + gl.placebo_years[year] + "_2"] = (
            data_copy["rel_placebo_cohort" + gl.placebo_years[year]] ** 2
        )

    return data_copy


def create_relative_month(data):
    """Create a variable indicating the distance to the pivotal month in months.

    This variable is for the 10-years-window treatment indicator. In addition, a
    squared term is created.

    Args:
        data (pandas DataFrame): The data set.

    Returns:
        data_copy (pandas DataFrame): The data with the additional column.

    """
    data_copy = data.copy()

    data_copy["rel_month"] = np.where(
        _treatment_reform_condition_w_month(data=data_copy, years=10, n_reform=1),
        -12 * (data_copy["pivotal_lower_reform1"] - 1 - data_copy["brth_year"])
        - (data_copy["pivotal_month_reform1"] - data_copy["brth_month"]),
        np.where(
            _treatment_reform_condition_w_month(data=data_copy, years=10, n_reform=2),
            -12 * (data_copy["pivotal_lower_reform2"] - 1 - data_copy["brth_year"])
            - (data_copy["pivotal_month_reform2"] - data_copy["brth_month"]),
            np.where(
                _control_reform_condition_w_month(data=data_copy, years=10, n_reform=1),
                -12 * (data_copy["pivotal_lower_reform1"] - 1 - data_copy["brth_year"])
                - (data_copy["pivotal_month_reform1"] - data_copy["brth_month"]),
                np.where(
                    _control_reform_condition_w_month(
                        data=data_copy,
                        years=10,
                        n_reform=2,
                    ),
                    -12
                    * (data_copy["pivotal_lower_reform2"] - 1 - data_copy["brth_year"])
                    - (data_copy["pivotal_month_reform2"] - data_copy["brth_month"]),
                    np.nan,
                ),
            ),
        ),
    )

    data_copy["rel_month2"] = data_copy["rel_month"] ** 2

    return data_copy


def _treatment_reform_condition(data, years, n_reform, placebo=0):
    """Get the condition for identifying treated individuals.

    Args:
        data (pandas DataFrame): The data set.
        years (int): Number of years to be included before and after the pivotal cohort.
        n_reform (int): 1 or 2; Which reform to look at.
        placebo (int): The distance to the true pivotal cohort.

    Returns:
        condition (pandas Series): A series with boolean values.

    """
    condition = (
        data["brth_year"] >= (data["pivotal_lower_reform" + str(n_reform)] + placebo)
    ) & (
        data["brth_year"]
        < (data["pivotal_lower_reform" + str(n_reform)] + years + placebo)
    )

    return condition


def _control_reform_condition(data, years, n_reform, placebo=0):
    """Get the condition for identifying control individuals.

    Args:
        data (pandas DataFrame): The data set.
        years (int): Number of years to be included before and after the pivotal cohort.
        n_reform (int): 1 or 2; Which reform to look at.
        placebo (int): The distance to the true pivotal cohort.

    Returns:
        condition (pandas Series): A series with boolean values.

    """
    condition = (
        data["brth_year"] < data["pivotal_lower_reform" + str(n_reform)] + placebo
    ) & (
        data["brth_year"]
        >= (data["pivotal_lower_reform" + str(n_reform)] - years + placebo)
    )

    return condition


def _treatment_reform_condition_w_month(data, years, n_reform):
    """Get the condition for identifying treated individuals also based on birth month.

    For instance the cut-off is March 31. Then individuals who are six by March 31
    enter school already, but those born later in that year enter school one year later.
    Thus, the first treated school cohort consists of individuals born by March 31 in the
    "pivotal year" or the year before after March 31.

    Args:
        data (pandas DataFrame): The data set.
        years (int): Number of years to be included before and after the pivotal cohort.
        n_reform (int): 1 or 2; Which reform to look at.

    Returns:
        condition (pandas Series): A series with boolean values.

    """
    condition = (
        (
            # born in or after pivotal year and before pivotal year + years -1
            (data["brth_year"] >= data["pivotal_lower_reform" + str(n_reform)])
            & (
                data["brth_year"]
                < data["pivotal_lower_reform" + str(n_reform)] + years - 1
            )
        )
        | (
            # born in pivotal year + years -1: check month
            (
                data["brth_year"]
                == data["pivotal_lower_reform" + str(n_reform)] + years - 1
            )
            & (data["brth_month"] < data["pivotal_month_reform" + str(n_reform)])
        )
        | (
            # born in pivotal year -1: check month
            (data["brth_year"] == data["pivotal_lower_reform" + str(n_reform)] - 1)
            & (data["brth_month"] >= data["pivotal_month_reform" + str(n_reform)])
        )
    )

    return condition


def _control_reform_condition_w_month(data, years, n_reform):
    """Get the condition for identifying treated individuals also based on birth month.

    For instance the cut-off is March 31. Then individuals who are six by March 31
    enter school already, but those born later in that year enter school one year later.
    Thus, the last control school cohort consists of individuals born up to March 31 in the
    year prior to the "pivotal year".

    Args:
        data (pandas DataFrame): The data set.
        years (int): Number of years to be included before and after the pivotal cohort.
        n_reform (int): 1 or 2; Which reform to look at.

    Returns:
        condition (pandas Series): A series with boolean values.

    """
    condition = (
        (
            # born before pivotal year -1 and after pivotal year - years
            (data["brth_year"] < data["pivotal_lower_reform" + str(n_reform)] - 1)
            & (
                data["brth_year"]
                >= data["pivotal_lower_reform" + str(n_reform)] - years
            )
        )
        | (
            # born in pivotal year - years -1: check month
            (
                data["brth_year"]
                == data["pivotal_lower_reform" + str(n_reform)] - years - 1
            )
            & (data["brth_month"] >= data["pivotal_month_reform" + str(n_reform)])
        )
        | (
            # born in pivotal year -1: check month
            (data["brth_year"] == data["pivotal_lower_reform" + str(n_reform)] - 1)
            & (data["brth_month"] < data["pivotal_month_reform" + str(n_reform)])
        )
    )

    return condition
