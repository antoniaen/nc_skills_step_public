"""Plot-Functions."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_years_of_education(data, staggered, not_staggered):
    """Plot years of education by country and reform.

    Args:
        data (pandas DataFrame): The data.
        staggered (list): List of staggered reforms.
        not_staggered (list): List of non-staggered reforms.

    Returns:
        fig (plotly.graph_objects.Figure): The figure.

    """
    plot_data_not_staggered = data[
        data["country_reform"].isin(not_staggered)
    ].copy()
    plot_data_staggered = data[data["country_reform"].isin(staggered)].copy()

    plot_data_not_staggered["country_reform"] = plot_data_not_staggered[
        "country_reform"
    ].replace(
        {
            "Bolivia1994": "Bolivia 1994",
            "Colombia1991": "Colombia 1991",
            "Ghana1961": "Ghana 1961",
            "Ghana1987": "Ghana 1987",
        },
    )
    plot_data_staggered["country_reform"] = plot_data_staggered["country_reform"].replace(
        {"Vietnam1991": "Vietnam 1991"},
    )

    plot_data_not_staggered = plot_data_not_staggered.groupby(
        ["country_reform", "brth_year"],
    )[["years_educ"]].mean()
    plot_data_staggered = plot_data_staggered.groupby(["country_reform", "brth_year"])[
        ["years_educ"]
    ].mean()

    fig_not_staggered = px.scatter(
        plot_data_not_staggered.reset_index(),
        x="brth_year",
        y="years_educ",
        color="country_reform",
        color_discrete_map={
            "Bolivia 1994": "yellowgreen",
            "Colombia 1991": "olive",
            "Ghana 1961": "olivedrab",
            "Ghana 1987": "forestgreen",
        },
    )
    fig_not_staggered.update_traces(marker={"symbol": "circle-open"})

    fig_staggered = px.scatter(
        plot_data_staggered.reset_index(),
        x="brth_year",
        y="years_educ",
        color="country_reform",
        color_discrete_map={
            "Vietnam 1991": "mediumvioletred",
        },
    )
    fig_staggered.update_traces(marker={"symbol": "circle-open"})

    fig = go.Figure(data=fig_not_staggered.data + fig_staggered.data)

    # Update the layout to move the legend below the plot
    fig.update_layout(
        showlegend=False,
        xaxis_title="Year of birth",
        yaxis_title="Average years of education",
        margin={"l": 50, "r": 50, "t": 40, "b": 40},  # Adjust margins for better layout
    )

    fig.add_vline(
        x=1983 - 0.5,
        line_dash="dash",
        line_width=1,
        line_color="yellowgreen",
    )
    fig.add_vline(x=1981 - 0.5, line_dash="dash", line_width=1, line_color="olive")
    fig.add_vline(x=1954 - 0.5, line_dash="dash", line_width=1, line_color="olivedrab")
    fig.add_vline(
        x=1975 - 0.5,
        line_dash="dash",
        line_width=1,
        line_color="forestgreen",
    )
    fig.add_vline(
        x=1977 - 0.5,
        line_dash="dash",
        line_width=1,
        line_color="mediumvioletred",
    )

    fig.add_annotation(
        x=1954 - 0.5,
        y=11.5,
        text="Birth year cutoff",
        showarrow=True,
        arrowhead=1,
    )

    fig.add_annotation(
        x=1985,
        y=5.5,
        text="Bolivia 1994",
        showarrow=False,
        font={"color": "yellowgreen"},
    )
    fig.add_annotation(
        x=1983.5,
        y=9.5,
        text="Colombia 1991",
        showarrow=False,
        font={"color": "olive"},
    )
    fig.add_annotation(
        x=1956,
        y=5.5,
        text="Ghana 1961",
        showarrow=False,
        font={"color": "olivedrab"},
    )
    fig.add_annotation(
        x=1972,
        y=5.5,
        text="Ghana 1987",
        showarrow=False,
        font={"color": "forestgreen"},
    )
    fig.add_annotation(
        x=1973.5,
        y=9.5,
        text="Vietnam 1991",
        showarrow=False,
        font={"color": "mediumvioletred"},
    )

    return fig


def plot_outcomes(data, outcome, reg_results, nice_variable_names, months=False):
    """Plot years of education by country and reform.

    Args:
        data (pandas DataFrame): The data.
        outcome (str): Outcome variable to plot.
        reg_results (OLSResults): Regression results.
        nice_variable_names (dict): Dictionary with nice variable names.
        months (bool): If True: months instead of years are plotted on the x-axis.

    Returns:
        fig (plotly.graph_objects.Figure): The figure.

    """
    if months is False:
        reform_indicator = "country_reform"
        x_axis_variable = "brth_year"
        treatment_indicator = "treated"

    elif months is True:
        reform_indicator = "country_reform_w_month"
        x_axis_variable = "month_bins_median"
        treatment_indicator = "treated_w_month"
        data["month_bins"] = pd.cut(data["rel_month"], 16)
        data["month_bins_median"] = data.groupby("month_bins")["rel_month"].transform(
            "median",
        )

    nice_variable_names["years_educ_calc"] = "Years spent in education"

    plot_data = data.copy()

    # Predict values.
    plot_data["predicted_values"] = reg_results.predict(plot_data)

    plot_data[reform_indicator] = plot_data[reform_indicator].replace(
        {
            "Ghana1961": "Ghana 1961",
            "Ghana1987": "Ghana 1987",
            "Vietnam1991": "Vietnam 1991",
            "Colombia1991": "Colombia 1991",
            "Bolivia1994": "Bolivia 1994",
        },
    )

    plot_data = plot_data.groupby(
        [reform_indicator, x_axis_variable],
    )[[outcome, "predicted_values", treatment_indicator, "partially_treated"]].mean()

    color_discrete_map = {
        "Ghana 1961": "steelblue",
        "Ghana 1987": "lightsteelblue",
        "Vietnam 1991": "mediumslateblue",
        "Colombia 1991": "blue",
        "Bolivia 1994": "deepskyblue",
    }

    fig1 = px.scatter(
        plot_data.reset_index(),
        x=x_axis_variable,
        y=outcome,
        color=reform_indicator,
        color_discrete_map=color_discrete_map,
    )

    fig2 = px.line(
        plot_data.reset_index().query(treatment_indicator + " == 0"),
        x=x_axis_variable,
        y="predicted_values",
        color=reform_indicator,
        color_discrete_map=color_discrete_map,
    )
    fig3 = px.line(
        plot_data.reset_index().query(
            treatment_indicator + " == 1 & " + reform_indicator + " != 'Vietnam 1991'",
        ),
        x=x_axis_variable,
        y="predicted_values",
        color=reform_indicator,
        color_discrete_map=color_discrete_map,
    )
    fig4 = px.line(
        plot_data.reset_index().query("partially_treated == 1"),
        x=x_axis_variable,
        y="predicted_values",
        color=reform_indicator,
        color_discrete_map=color_discrete_map,
    )

    fig = go.Figure(data=fig1.data + fig2.data + fig3.data + fig4.data)

    # Update the layout to move the legend below the plot
    fig.update_layout(
        legend={
            "orientation": "h",  # Set the orientation to horizontal
            "y": -0.15,  # Set the y position to move the legend below the plot
        },
        xaxis_title="Birth year",
        yaxis_title="Average of " + nice_variable_names[outcome],
        margin={"l": 50, "r": 50, "t": 40, "b": 40},  # Adjust margins for better layout
        font_family="Computer Modern Roman",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_traces(line={"width": 0.5})

    if months is False:
        fig.add_vline(
            x=1983 - 0.5,
            line_dash="dash",
            line_color=color_discrete_map["Bolivia 1994"],
        )
        fig.add_vline(
            x=1981 - 0.5,
            line_dash="dash",
            line_color=color_discrete_map["Colombia 1991"],
        )
        fig.add_vline(
            x=1954 - 0.5,
            line_dash="dash",
            line_color=color_discrete_map["Ghana 1961"],
        )
        fig.add_vline(
            x=1975 - 0.5,
            line_dash="dash",
            line_color=color_discrete_map["Ghana 1987"],
        )
        fig.add_vline(
            x=1977 - 0.5,
            line_dash="dash",
            line_color=color_discrete_map["Vietnam 1991"],
        )

    else:
        pass

    return fig


def coeff_plot_two_xaxis(results_list, nice_variable_names, group_mapping):
    """Plot point estimates and confidence intervals of OLSResults.

    Args:
        results_list (list): List of OLSResults.
        nice_variable_names (dict): Dictionary with nice variable names.
        group_mapping (dict): Dictionary mapping dependent variables to groups ("Personality and behavior" or "Preferences").

    Returns:
        fig (plotly.graph_objects.Figure): The figure.

    """
    coeff_sizes = []
    dependent_vars = []
    conf_intervals_errors = []

    # Extract coefficient estimates, dependent variables,
    # and "errors" based on confidence intervals.
    for result in results_list:
        coeff_sizes.append(result.params["treated"])
        dependent_vars.append(result.model.endog_names)
        conf_intervals_errors.append(
            result.conf_int().loc["treated", 1] - result.params["treated"],
        )

    # Replace dependent_vars with nice names.
    nice_names = [nice_variable_names[var] for var in dependent_vars]

    # Convert the lists into a DataFrame for easier manipulation.
    plot_data = pd.DataFrame(
        data={
            "coeff_sizes": coeff_sizes,
            "conf_intervals_errors": conf_intervals_errors,
            "nice_names": nice_names,
        },
    )

    # Add a column indicating on which axis the variable should be plotted.
    plot_data["type"] = plot_data["nice_names"].map(group_mapping)

    # Get scale factor to align the second x-axis.
    x1_min = (
        plot_data.query("type=='Personality and behavior'")["coeff_sizes"]
        - plot_data.query("type=='Personality and behavior'")["conf_intervals_errors"]
    ).min()
    x1_max = (
        plot_data.query("type=='Personality and behavior'")["coeff_sizes"]
        + plot_data.query("type=='Personality and behavior'")["conf_intervals_errors"]
    ).max()
    x2_min = (
        plot_data.query("type=='Preferences'")["coeff_sizes"]
        - plot_data.query("type=='Preferences'")["conf_intervals_errors"]
    ).min()
    x2_max = (
        plot_data.query("type=='Preferences'")["coeff_sizes"]
        + plot_data.query("type=='Preferences'")["conf_intervals_errors"]
    ).max()
    scale_factor = (x1_max - x1_min) / (x2_max - x2_min)

    layout = go.Layout(
        title="Point estimates and confidence intervals",
        xaxis=go.layout.XAxis(
            title="Personality and behavior (standardized)",
            range=[x1_min * 1.8, x1_max * 1.8],
        ),
        xaxis2=go.layout.XAxis(
            title="Preferences (binary)",
            range=[x1_min * 1.8 / scale_factor, x1_max * 1.8 / scale_factor],
            overlaying="x",
            side="top",
        ),
        yaxis={"title": "Dependent variables"},
    )

    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Scatter(
            x=plot_data.query("type=='Personality and behavior'")["coeff_sizes"],
            y=plot_data.query("type=='Personality and behavior'")["nice_names"],
            name="Personality and behavior",
            mode="markers",
            marker_color="darkblue",
            error_x={
                "type": "data",  # value of error bar given in data coordinates
                "array": plot_data.query("type=='Personality and behavior'")[
                    "conf_intervals_errors"
                ],
                "width": 0,
                "visible": True,
            },
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=plot_data.query("type=='Preferences'")["coeff_sizes"],
            y=plot_data.query("type=='Preferences'")["nice_names"],
            name="Preferences",
            # Plot on second x-axis.
            xaxis="x2",
            mode="markers",
            marker_color="lightblue",
            error_x={
                "type": "data",  # value of error bar given in data coordinates
                "array": plot_data.query("type=='Preferences'")[
                    "conf_intervals_errors"
                ],
                "width": 0,
                "visible": True,
            },
        ),
    )

    # Set line_width to adjust the thickness of the line.
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1)

    # Adjust the position of the figure title
    fig.update_layout(
        margin={"t": 140},  # Adjust this value to move the title up or down
        legend={
            "traceorder": "reversed",  # Set to 'normal' to maintain the order in which traces were added
        },
        font_family="Computer Modern Roman",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False)

    return fig


def hist_occupations(data, occ_type):
    """Histogram with occupations.

    Args:
        data (pandas DataFrame): The data.
        occ_type (str): Type of occupation categories.

    Returns:
        fig (plotly.graph_objects.Figure): The figure.

    """
    occ_types = {
        "occupation": [
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
        ],
        "occtype_step": [
            "Highly skilled white collar - Managers/Professionals/Technicians",
            "Low skilled white collar",
            "Crafts and related trades workers; Plant and machine operator and assemblers",
            "Elementary occupations",
            "Skilled agriculture work",
            "Military personnel",
        ],
    }
    data["treated"] = data["treated"].replace({0: "Control", 1: "Treated"})

    fig = px.histogram(
        data,
        x=occ_type,
        color="treated",
        histnorm="percent",
        barmode="group",
        category_orders={occ_type: occ_types[occ_type]},
        color_discrete_map={"Control": "mediumblue", "Treated": "gold"},
    )
    fig.update_layout(
        font_family="Computer Modern Roman",
        xaxis_title="Occupation",
        yaxis_title="Percent",
        legend_title=None,
        margin={
            "l": 100,
            "r": 20,
            "t": 40,
            "b": 40,
        },  # Adjust margins for better layout
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    return fig
