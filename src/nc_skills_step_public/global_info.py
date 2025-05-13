"""Store lists, dictionaries, etc.

that are used repeatedly and throughout the project.

"""
########### REFORMS ############
reforms_final = [
    "Ghana1961",
    "Ghana1987",
    "Colombia1991",
    "Vietnam1991",
    "Bolivia1994",
]

######### DEPENDENT VARIABLES ########
dependent_variables = [
    "years_educ",
    "years_educ_calc",
    "ln_earnings_h_usd",
    "emp",
    "wage_worker",
    "read_s",
    "write_s",
    "num_s",
    "extraversion_av_s_abcorr",
    "conscientiousness_av_s_abcorr",
    "openness_av_s_abcorr",
    "stability_av_s_abcorr",
    "agreeableness_av_s_abcorr",
    "grit_av_s_abcorr",
    "decision_av_s_abcorr",
    "hostile_av_s_abcorr",
    "risk_s",
    "risk_binary",
    "discount_s",
    "patience_binary",
]

groups_of_dependent_variables = {
    "years_educ": ["years_educ"],
    "wage": ["ln_earnings_h_usd"],
    "emp": ["emp"],
    "wage_worker": ["wage_worker"],
    "cogn_skills": ["num_s", "read_s", "write_s"],
    "ncogn_skills": [
        "agreeableness_av_s_abcorr",
        "conscientiousness_av_s_abcorr",
        "stability_av_s_abcorr",
        "extraversion_av_s_abcorr",
        "openness_av_s_abcorr",
        "decision_av_s_abcorr",
        "grit_av_s_abcorr",
        "hostile_av_s_abcorr",
    ],
    "risk": ["risk_s"],
    "time1": ["discount_s"],
    "preferences_binary": ["patience_binary", "risk_binary"],
}

nice_variable_names = {
    "years_educ": "Years of education",
    "ln_earnings_h_usd": "ln(wage) in USD",
    "emp": "Currently working",
    "wage_worker": "Wage worker vs self/family-employed",
    "read_s": "Reading",
    "write_s": "Writing",
    "num_s": "Numeracy",
    "extraversion_av_s_abcorr": "Extraversion",
    "conscientiousness_av_s_abcorr": "Conscientiousness",
    "openness_av_s_abcorr": "Openness to experience",
    "stability_av_s_abcorr": "Emotional stability",
    "agreeableness_av_s_abcorr": "Agreeableness",
    "grit_av_s_abcorr": "Grit",
    "decision_av_s_abcorr": "Decision-making patterns",
    "hostile_av_s_abcorr": "Hostile attribution bias",
    "risk_s": "Willingness to take risk (non-binary)",
    "risk_binary": "Willingness to take risk",
    "discount_s": "Discount (non-binary)",
    "patience_binary": "Patience",
}

nice_variable_names_to_groups_mapping = {
    "Years of education": "Years of education",
    "ln(wage) in USD": "Labor market outcomes",
    "Currently working": "Labor market outcomes",
    "Reading": "Cognitive skills",
    "Writing": "Cognitive skills",
    "Numeracy": "Cognitive skills",
    "Extraversion": "Personality traits",
    "Conscientiousness": "Personality traits",
    "Openness to experience": "Personality traits",
    "Emotional stability": "Personality traits",
    "Agreeableness": "Personality traits",
    "Grit": "Behavior",
    "Decision-making patterns": "Behavior",
    "Hostile attribution bias": "Behavior",
    "Willingness to take risk (non-binary)": "Preferences",
    "Willingness to take risk": "Preferences",
    "Discount (non-binary)": "Preferences",
    "Patience": "Preferences",
}

nice_variable_names_to_broad_groups_mapping = {
    "Extraversion": "Personality and behavior",
    "Conscientiousness": "Personality and behavior",
    "Openness to experience": "Personality and behavior",
    "Emotional stability": "Personality and behavior",
    "Agreeableness": "Personality and behavior",
    "Grit": "Personality and behavior",
    "Decision-making patterns": "Personality and behavior",
    "Hostile attribution bias": "Personality and behavior",
    "Willingness to take risk": "Preferences",
    "Patience": "Preferences",
}

PCA_dependent_variables = [
    "extraversion_pca",
    "conscientiousness_pca",
    "openness_pca",
    "stability_pca",
    "agreeableness_pca",
    "grit_pca",
    "decision_pca",
    "hostile_pca",
]

placebo_years = {-6: "0", -5: "1", 5: "2", 6: "3", 7: "4"}
