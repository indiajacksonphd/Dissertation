import os
import boto3
from io import BytesIO
from solar_survival.analysis_functions import *


def welcome_message():
    current_directory = os.getcwd()
    greeting = "Welcome to Solar Survival!"

    title = "Advancing Solar Energetic Particle Event Prediction through\n"
    title += "Comprehensive Survival Analysis and Cloud Computing"

    name = "India Jackson\n"
    name += "Astrophysics Phd Candidate\n"
    name += "Georgia State University\n"
    name += "Atlanta, GA"

    abstract = "Solar energetic particles (SEPs) pose significant challenges due to their potential\n"
    abstract += "impact on technology, astronaut health, and space missions. This research employs\n"
    abstract += "an array of advanced statistical techniques and cloud computing resources to\n"
    abstract += "enhance the accuracy of SEP event predictions. Survival analysis is particularly\n"
    abstract += "highlighted for its capacity to handle censored data and offers insights into\n"
    abstract += "event probabilities over time. Our approach integrates a combination of\n"
    abstract += "non-parametric, semi-parametric, and parametric models including the Kaplan-\n"
    abstract += "Meier (KM) Estimator and Cox Proportional Hazards (Cox PH) Models alongside the\n"
    abstract += "exponential, Weibull, log-normal, and loglogistic distributions, along side. We\n"
    abstract += "bridge theoretical insights with practical applications, demonstrating the\n"
    abstract += "all-inclusive utility of these models for understanding and forecasting the\n"
    abstract += "dynamics of SEP events within the dynamic solar system. Leveraging Amazon Web\n"
    abstract += "Services (AWS) cloud-based computing resources and Python’s ”lifelines” and\n"
    abstract += "”scikit-survival” libraries, we preprocess and analyze our data remotely. This\n"
    abstract += "cloud based approach not only optimizes our current analysis but also\n"
    abstract += "establishes a foundation for future big data and machine learning endeavors. Our\n"
    abstract += "research aims to advance space weather prediction and to improve the resilience\n"
    abstract += "particle events.                                                                "

    folder_descriptions = [
        "datasets",
        "1_exploratory_analysis_all_events",
        "2_exploratory_analysis_classification",
        "3_km_estimation",
        "4_cox_proportional",
        "5_parametric_analysis"
    ]

    folder_info = "\nFolders:\n" + "\n".join(folder_descriptions)

    welcome_results = f"Your current directory is: {current_directory},\n"
    welcome_results += "this is where you can find the generated files and results!"
    welcome_results += folder_info


    all_lines = name.splitlines() + title.splitlines() + abstract.splitlines() + welcome_results.splitlines()
    max_line_length = max(len(line) for line in all_lines)

    border = "#" + "=" * (max_line_length + 2) + "#"
    welcome_box = border + "\n"
    welcome_box += "#" + greeting.center(max_line_length + 2) + "#\n"
    welcome_box += "#" + ' ' * (max_line_length + 2) + "#\n"

    for line in title.splitlines():
        centered_line = line.center(max_line_length + 2)
        welcome_box += "#" + centered_line + "#\n"

    welcome_box += "#" + ' ' * (max_line_length + 2) + "#\n"

    for line in name.splitlines():
        centered_line = line.center(max_line_length + 2)
        welcome_box += "#" + centered_line + "#\n"

    welcome_box += "#" + ' ' * (max_line_length + 2) + "#\n"

    for line in abstract.splitlines():
        centered_line = line.center(max_line_length + 2)
        welcome_box += "#" + centered_line + "#\n"

    welcome_box += "#" + ' ' * (max_line_length + 2) + "#\n"

    for line in welcome_results.splitlines():
        centered_line = line.center(max_line_length + 2)
        welcome_box += "#" + centered_line + "#\n"

    welcome_box += "#" + ' ' * (max_line_length + 2) + "#\n"
    welcome_box += border + "\n"
    print(welcome_box)


def data_extraction():
    # Define directory names
    directories = [
        'datasets',
        '1_exploratory_analysis_all_events',
        '1_exploratory_analysis_all_events/charts',
        '1_exploratory_analysis_all_events/data',
        '2_exploratory_analysis_classification',
        '2_exploratory_analysis_classification/charts',
        '2_exploratory_analysis_classification/data',
        '3_km_estimation',
        '3_km_estimation/curves',
        '3_km_estimation/life_tables',
        '3_km_estimation/life_tables/data',
        '3_km_estimation/life_tables/tables',
        '3_km_estimation/multivariate_log_rank',
        '3_km_estimation/multivariate_log_rank/data',
        '3_km_estimation/multivariate_log_rank/tables',
        '4_cox_proportional',
        '4_cox_proportional/data',
        '4_cox_proportional/graphs',
        '5_parametric_analysis',
        '5_parametric_analysis/curves',
        '5_parametric_analysis/data',
        '5_parametric_analysis/life_tables'
    ]

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    s3 = boto3.client('s3')

    bucket_name = 'my-dissertation'
    file_name = 'GSEP.csv'

    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    content = response['Body'].read()

    gsep_df = pd.read_csv(BytesIO(content), encoding='utf-8')

    # Remove numbers from the 'fl_goes_class' column
    gsep_df['fl_goes_class'] = gsep_df['fl_goes_class'].str.extract(r'([A-Za-z]+)')

    # Calculate the time difference between gsep_max_time and fl_start_time in minutes
    date_format = '%m/%d/%y %H:%M'
    gsep_df['time'] = (pd.to_datetime(gsep_df['gsep_max_time'], format=date_format) - pd.to_datetime(gsep_df['fl_start_time'], format=date_format)).dt.total_seconds() / 60

    # Create event columns based on the conditions
    gsep_df['event10'] = gsep_df['ppf_gt10MeV'].apply(lambda x: 1 if pd.notna(x) else 0)
    gsep_df['event30'] = gsep_df['ppf_gt30MeV'].apply(lambda x: 1 if pd.notna(x) else 0)
    gsep_df['event60'] = gsep_df['ppf_gt60MeV'].apply(lambda x: 1 if pd.notna(x) else 0)
    gsep_df['event100'] = gsep_df['ppf_gt100MeV'].apply(lambda x: 1 if pd.notna(x) else 0)

    # Select the desired columns for SSEP dataset
    sse_df = gsep_df[['time', 'event10', 'event30', 'event60', 'event100', 'fl_lon', 'fl_lat', 'fl_goes_class']]
    sse_df_sorted = sse_df.sort_values(by='time', ascending=True)

    # Sort the DataFrame based on the 'time' column in ascending order

    sse_df_sorted.to_csv('datasets/SSEP_missing_values.csv', index=False)

    # Drop rows with any missing values (NaN) in any cell
    sse_df_2 = sse_df_sorted.copy().dropna(axis=0, how='any')
    sse_df_2.to_csv('datasets/SSEP_cleaned.csv', index=False)

    # Load the SSEP dataset from the CSV file
    sse_file_path = 'datasets/SSEP_cleaned.csv'  # Replace with the actual path of your cleaned SSEP dataset
    sse_df = pd.read_csv(sse_file_path)
    return sse_df


def results():
    sse_file_path = 'datasets/SSEP_cleaned.csv'
    sse_df = pd.read_csv(sse_file_path)

    # 1) Histogram of entire dataset
    histogram_all(sse_df, 500)

    # 2) Boxplot and Violin plot for time
    box_plot_all(sse_df)

    # 3) Pie Chart for group counts
    pie_chart_classification(sse_df)

    # 4) Violin Plots by Classification
    box_plot_classification(sse_df)

    # 5) Survival curves for >10, >30, and >60 threshold for each class (C, M, X)
    plot_survival_curves(sse_df, [10, 30, 60, 100])

    # 6) Summary statistics for entire dataset
    summary_entire_set = sse_df['time'].describe()
    save_summary_to_table(summary_entire_set, '1_exploratory_analysis_all_events/data/summary_entire_set.csv')
    save_data_summary_table(summary_entire_set, '1_exploratory_analysis_all_events/data/data_summary.png')

    # 7) Frequency table for entire set
    frequency_table = pd.cut(sse_df['time'], bins=range(0, int(sse_df['time'].max()) + 501, 500), right=False).value_counts()
    save_frequency_table(frequency_table, table_title='Frequency Table', filename='1_exploratory_analysis_all_events/data/frequency_table.png')
    save_summary_to_table(frequency_table, '1_exploratory_analysis_all_events/data/frequency_table.csv')

    # 8) Summary statistics by flare classification
    summary_by_classification = sse_df.groupby('fl_goes_class')['time'].describe()
    save_data_summary_table_by_classification(summary_by_classification, '2_exploratory_analysis_classification/data/data_summary_classification.png')
    save_summary_to_table(summary_by_classification, '2_exploratory_analysis_classification/data/summary_by_classification.csv')

    #####################################################################
    # 10) Cox PH and Schoenfeld Test
    # print('the summary of the Cox PH model for >60 threshold')
    fit_coxph_model(sse_df, 60)
    # print('the summary of the Cox PH model for >100 threshold')
    fit_coxph_model(sse_df, 100)

    #####################################################################
    # 11) Cox PH, Schoenfeld Graph and Proportional Hazards Assumption
    proportional_assumption(sse_df, 60)
    proportional_assumption(sse_df, 100)

    #####################################################################
    # 12) Survival Probability Models

    parametric_models = {
        'exponential': ExponentialFitter(),
        'weibull': WeibullFitter(),
        'lognormal': LogNormalFitter(),
        'loglogistic': LogLogisticFitter(),
    }

    create_parametric_model_table(sse_df, [60], parametric_models)
    create_parametric_model_table(sse_df, [100], parametric_models)


if __name__ == '__main__':
    welcome_message()
    data_extraction()  # Call data extraction first
    results()  # Then call your wrapper function
