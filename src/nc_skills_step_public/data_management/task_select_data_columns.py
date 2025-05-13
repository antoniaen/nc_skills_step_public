"""Select columns from each country dataset and store smaller datasets."""

import pandas as pd
import pytask

from nc_skills_step_public.config import BLD, SRC

countries_2012 = [
    "Bolivia",
    "Colombia",
    "Vietnam",
]

countries_2013 = [
    "Ghana",
]

for country in countries_2012 + countries_2013:
    kwargs = {
        "country": country,
        "depends_on": {"data": SRC / "data" / f"STEP {country}_working.dta"},
        "produces": BLD / "python" / "data" / f"{country}_small.pkl",
    }

    @pytask.mark.depends_on(
        {
            "selected_variables": SRC / "data_management" / "selected_variables.xlsx",
        },
    )
    @pytask.mark.task(id=country, kwargs=kwargs)
    def task_select_data_columns(depends_on, country, produces):
        """Create a smaller dataset with selected columns only."""
        selected_variables = pd.read_excel(depends_on["selected_variables"])
        sel_vars_list_2012 = selected_variables["Name"].to_list()

        # Add plausible values for the literacy test.
        sel_vars_list_2012 += [
            "PVLIT1",
            "PVLIT2",
            "PVLIT3",
            "PVLIT4",
            "PVLIT5",
            "PVLIT6",
            "PVLIT7",
            "PVLIT8",
            "PVLIT9",
            "PVLIT10",
        ]

        # Experience of violence or abuse as a child is only available for Bolivia.
        if country == "Bolivia":
            sel_vars_list_2012 += ["m7a_q27", "m7a_q28"]

        # Some variable names for 2013 differ.
        var_names_mapping_12_to_13 = {
            "m2_q29": "m2_q26",
            "m7a_q23": "m7_q22",
            "m7a_q25": "m7_q24",
        }

        sel_vars_list_2013 = []
        for item in sel_vars_list_2012:
            if item.startswith("m6a_q01"):
                parts = item.split("m6a_q01")
                new_item = "m6a_q01_" + parts[1]
                sel_vars_list_2013.append(new_item)

            elif item in var_names_mapping_12_to_13:
                sel_vars_list_2013.append(var_names_mapping_12_to_13[item])

            else:
                sel_vars_list_2013.append(item)

        var_names_mapping_trait_items = dict(
            zip(
                [name for name in sel_vars_list_2012 if name.startswith("m6a_q01")],
                [name for name in sel_vars_list_2013 if name.startswith("m6a_q01")],
            ),
        )

        var_names_mapping_12_to_13.update(var_names_mapping_trait_items)

        data = pd.read_stata(depends_on["data"], convert_categoricals=False)

        if country in countries_2012:
            data_sel_vars_only = data[sel_vars_list_2012]

        elif country in countries_2013:
            data_sel_vars_only = data[sel_vars_list_2013]
            data_sel_vars_only = data_sel_vars_only.rename(
                columns={
                    value: key for key, value in var_names_mapping_12_to_13.items()
                },
            )

        data_sel_vars_only["country"] = country

        data_sel_vars_only.to_pickle(produces)