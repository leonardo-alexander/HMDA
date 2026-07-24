"""Shared constants for the HMDA KDD pipeline (Phase 1-5).

Extracted from HMDA.ipynb so Phase 1-4 modules and the dashboard's
build_data.py share one definition instead of duplicating it.
"""

RANDOM_STATE = 42

HF_URL = "https://huggingface.co/datasets/leonardo-alexander/hmda_sample/resolve/main/hmda_sample.csv"

# --- Phase 1: column-role segmentation (must partition every raw column) ---

CONTINUOUS = [
    "loan_amount", "combined_loan_to_value_ratio", "interest_rate", "rate_spread",
    "total_loan_costs", "total_points_and_fees", "origination_charges", "discount_points",
    "lender_credits", "loan_term", "prepayment_penalty_term", "intro_rate_period",
    "property_value", "income", "multifamily_affordable_units",
    "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage", "tract_owner_occupied_units",
    "tract_one_to_four_family_homes", "tract_median_age_of_housing_units",
]

STRING_BAND = ["debt_to_income_ratio", "applicant_age", "co_applicant_age", "total_units"]

CATEG_CODE = [
    "action_taken", "purchaser_type", "preapproval", "loan_type", "loan_purpose", "lien_status",
    "reverse_mortgage", "open_end_line_of_credit", "business_or_commercial_purpose", "hoepa_status",
    "negative_amortization", "interest_only_payment", "balloon_payment", "other_nonamortizing_features",
    "construction_method", "occupancy_type",
    "manufactured_home_secured_property_type", "manufactured_home_land_property_interest",
    "applicant_credit_score_type", "co_applicant_credit_score_type",
    "submission_of_application", "initially_payable_to_institution",
]

TEXT_CATEG = [
    "state_code", "conforming_loan_limit", "derived_loan_product_type", "derived_dwelling_category",
    "derived_ethnicity", "derived_race", "derived_sex",
]

IDS = ["activity_year", "lei", "derived_msa_md", "county_code", "census_tract"]

DEMOGRAPHIC_RAW = [
    "applicant_ethnicity_1", "applicant_ethnicity_2", "applicant_ethnicity_3",
    "applicant_ethnicity_4", "applicant_ethnicity_5",
    "co_applicant_ethnicity_1", "co_applicant_ethnicity_2", "co_applicant_ethnicity_3",
    "co_applicant_ethnicity_4", "co_applicant_ethnicity_5",
    "applicant_race_1", "applicant_race_2", "applicant_race_3",
    "applicant_race_4", "applicant_race_5",
    "co_applicant_race_1", "co_applicant_race_2", "co_applicant_race_3",
    "co_applicant_race_4", "co_applicant_race_5",
    "applicant_sex", "co_applicant_sex",
    "applicant_ethnicity_observed", "co_applicant_ethnicity_observed",
    "applicant_race_observed", "co_applicant_race_observed",
    "applicant_sex_observed", "co_applicant_sex_observed",
    "applicant_age_above_62", "co_applicant_age_above_62",
]

AUS = ["aus_1", "aus_2", "aus_3", "aus_4", "aus_5"]
DENIAL = ["denial_reason_1", "denial_reason_2", "denial_reason_3", "denial_reason_4"]

SENTINEL_EXEMPT = 1111

GROUPS = {
    "CONTINUOUS": CONTINUOUS, "STRING_BAND": STRING_BAND, "CATEG_CODE": CATEG_CODE,
    "TEXT_CATEG": TEXT_CATEG, "IDS": IDS, "DEMOGRAPHIC_RAW": DEMOGRAPHIC_RAW,
    "AUS": AUS, "DENIAL": DENIAL,
}

LABELS = {
    "action_taken": {
        "1": "Originated", "2": "Approved_NotAccepted", "3": "Denied",
        "4": "Withdrawn", "5": "Incomplete", "6": "Purchased",
        "7": "Preapproval_Denied", "8": "Preapproval_Approved_NotAccepted",
    },
    "loan_type": {"1": "Conventional", "2": "FHA", "3": "VA", "4": "RHS_FSA"},
    "loan_purpose": {"1": "Home_Purchase", "2": "Home_Improvement", "31": "Refinance",
                     "32": "CashOut_Refinance", "4": "Other", "5": "NotApplicable"},
    "lien_status": {"1": "First_Lien", "2": "Subordinate_Lien"},
    "occupancy_type": {"1": "Principal_Residence", "2": "Second_Residence", "3": "Investment"},
    "construction_method": {"1": "Site_Built", "2": "Manufactured"},
    "hoepa_status": {"1": "High_Cost", "2": "Not_High_Cost", "3": "NotApplicable"},
    "preapproval": {"1": "Requested", "2": "Not_Requested"},
    "reverse_mortgage": {"1": "Yes", "2": "No", "1111": "Exempt"},
    "open_end_line_of_credit": {"1": "Yes", "2": "No", "1111": "Exempt"},
    "business_or_commercial_purpose": {"1": "Yes", "2": "No", "1111": "Exempt"},
    "applicant_credit_score_type": {
        "1": "EquifaxBeacon5", "2": "ExperianFairIsaac", "3": "FICOv4", "4": "FICOv98",
        "5": "VantageScore2", "6": "VantageScore3", "7": "MoreThanOne", "8": "Other",
        "9": "NotApplicable", "1111": "Exempt",
    },
}

SENTINEL_WHITELIST = {
    "1111": ["reverse_mortgage", "open_end_line_of_credit", "business_or_commercial_purpose",
             "negative_amortization", "interest_only_payment", "balloon_payment",
             "other_nonamortizing_features", "manufactured_home_secured_property_type",
             "manufactured_home_land_property_interest", "applicant_credit_score_type",
             "co_applicant_credit_score_type", "submission_of_application",
             "initially_payable_to_institution", "aus_1"],
    "8888": ["applicant_age", "co_applicant_age"],
    "9999": ["co_applicant_age"],
}

AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74", "Age_NA", "No_CoApplicant"]
DTI_ORDER = ["<20%", "20%-<30%", "30%-<36%", "36%-<43%", "43%-<50%", "50%-60%", ">60%", "Exempt"]
UNITS_ORDER = ["1", "2", "3", "4", "5-24", "25-49", "50-99", "100-149", ">149"]

MISSING_DROP_THRESHOLD = 0.60

bin_specs = {
    "income_band": ("income",
        [-float("inf"), 30, 50, 75, 100, 150, 200, float("inf")],
        ["<30k", "30-50k", "50-75k", "75-100k", "100-150k", "150-200k", ">200k"]),
    "loan_amount_band": ("loan_amount",
        [-float("inf"), 100_000, 200_000, 300_000, 500_000, 750_000, float("inf")],
        ["<100k", "100-200k", "200-300k", "300-500k", "500-750k", ">750k"]),
    "property_value_band": ("property_value",
        [-float("inf"), 100_000, 200_000, 350_000, 500_000, 750_000, float("inf")],
        ["<100k", "100-200k", "200-350k", "350-500k", "500-750k", ">750k"]),
    "interest_rate_band": ("interest_rate",
        [-float("inf"), 3, 4, 5, 6, 7, float("inf")],
        ["<3%", "3-4%", "4-5%", "5-6%", "6-7%", ">7%"]),
    "cltv_band": ("combined_loan_to_value_ratio",
        [-float("inf"), 60, 80, 90, 95, 100, float("inf")],
        ["<60%", "60-80%", "80-90%", "90-95%", "95-100%", ">100%"]),
    "tract_income_cat": ("tract_to_msa_income_percentage",
        [-float("inf"), 50, 80, 120, float("inf")],
        ["Low_Income", "Moderate_Income", "Middle_Income", "Upper_Income"]),
    "tract_minority_cat": ("tract_minority_population_percent",
        [-float("inf"), 20, 50, 80, float("inf")],
        ["Low_Minority", "Moderate_Minority", "High_Minority", "Majority_Minority"]),
}

# --- Phase 1: feature-selection leakage guard (post-decision pricing fields) ---

LEAKAGE = ["interest_rate", "rate_spread", "total_loan_costs", "total_points_and_fees",
           "origination_charges", "discount_points", "lender_credits", "hoepa_status",
           "purchaser_type", "interest_rate_was_missing", "rate_spread_was_missing",
           "total_loan_costs_was_missing", "origination_charges_was_missing"]
