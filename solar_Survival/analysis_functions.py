import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages
from lifelines.statistics import proportional_hazard_test
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import KaplanMeierFitter, ExponentialFitter, WeibullFitter, LogNormalFitter, LogLogisticFitter, CoxPHFitter


########################################### EXPLORATORY DATA ANALYSIS ########################################################
# Function to create the histogram of time in minutes for the entire dataset
def histogram_all(df, bin_width):
    plt.figure(figsize=(8, 6))

    # Calculate the number of bins based on the bin width and the range of data
    time_range = df['time'].max() - df['time'].min()
    num_bins = int(np.ceil(time_range / bin_width))

    bins = np.linspace(0, df['time'].max(), num_bins + 1)  # Specify bin edges starting from 0
    plt.hist(df['time'], bins=bins, edgecolor='black')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time in Minutes (Entire Dataset)')
    plt.savefig(f'1_exploratory_analysis_all_events/charts/histogram.png')  # Save the plot as a PNG file
    plt.close()


# Function to create Box and Violin plot for time
def box_plot_all(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='time', data=df, width=0.2, color='pink')  # Plot box plot
    plt.title('Box Plot for Time to Detection')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Survival Time')
    plt.savefig(f'1_exploratory_analysis_all_events/charts/box_plot.png')
    plt.close()


# Groups (Solar Flare Classification; C, M, X) analysis
# Function to create the pie chart of flare classification count
def pie_chart_classification(df):
    plt.figure(figsize=(6, 6))
    df['fl_goes_class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Pie Chart of Flare Classification Count')
    plt.ylabel('')
    plt.savefig(f'2_exploratory_analysis_classification/charts/pie_chart_classification.png')  # Save the plot as a PNG file
    plt.close()


def box_plot_classification(df):
    plt.figure(figsize=(10, 6))
    class_color = {'C': '#4F993C', 'M': '#3B75AF', 'X': '#EA8335'}
    sns.boxplot(y='fl_goes_class', x='time', data=df, width=0.2, palette=class_color)  # Plot box plot inside the violin plot
    plt.title('Box Plot of Time to Detection by Flare Classification')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Flare Classification')
    plt.savefig(f'2_exploratory_analysis_classification/charts/box_plot_by_classification.png')  # Save the plot as a PNG file
    plt.close()


def save_frequency_table(frequency_table, table_title, filename):
    # Convert the frequency table to a DataFrame
    df_frequency_table = pd.DataFrame({'Interval': frequency_table.index, 'Frequency': frequency_table.values})

    # Create a table visualization using Matplotlib
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')  # Hide axes
    ax.table(cellText=df_frequency_table.values, colLabels=df_frequency_table.columns, cellLoc='center', loc='upper left')

    # Add a title to the table
    ax.text(0.5, 1.1, table_title, fontsize=12, fontweight='bold', ha='center', va='center', transform=ax.transAxes)

    # Save the table visualization as a PNG file
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def save_data_summary_table(summary_entire_set, filename):
    # Create a dictionary with the data
    data = {
        'Total': [summary_entire_set['count']],
        'Min': [summary_entire_set['min']],
        'Median': [summary_entire_set['50%']],
        'Max': [summary_entire_set['max']]
    }

    # Convert the dictionary to a DataFrame
    df_summary = pd.DataFrame(data)

    # Add the table title
    table_title = 'Data Summary'

    # Use the tabulate function to format the DataFrame as a table
    tabulate(df_summary, headers='keys', tablefmt='grid')

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(6, 3))

    # Hide axes
    ax.axis('off')

    # Add the table to the axes
    ax.table(cellText=df_summary.values, colLabels=df_summary.columns, cellLoc='center', loc='upper left')

    # Add a title to the table using ax.text
    ax.text(0.5, 1.1, table_title, fontsize=12, fontweight='bold', ha='center', va='center')

    # Save the table visualization as a PNG file
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def save_data_summary_table_by_classification(summary_by_classification, filename):
    headers = ['Classification', 'Total', 'Min', 'Median', 'Max']
    rows = [
        ['C', summary_by_classification.loc['C', 'count'], summary_by_classification.loc['C', 'min'],
         summary_by_classification.loc['C', '50%'], summary_by_classification.loc['C', 'max']],
        ['M', summary_by_classification.loc['M', 'count'], summary_by_classification.loc['M', 'min'],
         summary_by_classification.loc['M', '50%'], summary_by_classification.loc['M', 'max']],
        ['X', summary_by_classification.loc['X', 'count'], summary_by_classification.loc['X', 'min'],
         summary_by_classification.loc['X', '50%'], summary_by_classification.loc['X', 'max']]
    ]
    tabulate(rows, headers=headers, tablefmt='grid')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')  # Hide axes
    ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='upper left')

    # Add title to the table
    table_title = 'Data Summary Classification'
    ax.text(0.5, 1.1, table_title, fontsize=12, fontweight='bold', ha='center', va='center', transform=ax.transAxes)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def save_summary_to_table(summary_data, filename):
    df = pd.DataFrame(summary_data)
    df.to_csv(filename, index=False)

################################################# KAPLAN-MEIER ESTIMATION ####################################################


# Helper function to perform the log-rank test for a single threshold
def perform_logrank_test(df, threshold):
    kmf = KaplanMeierFitter()
    groups = df['fl_goes_class'].unique()
    survival_dict = {}
    results = []  # List to store the results

    for group in groups:
        group_df = df[df['fl_goes_class'] == group]
        event_observed = group_df[f'event_{threshold}'] if f'event_{threshold}' in group_df.columns else group_df['event100']
        time = group_df['time']
        kmf.fit(time, event_observed=event_observed, label=group)
        survival_dict[group] = kmf.survival_function_

        # Perform log-rank test
        other_groups = [g for g in groups if g != group]
        for other_group in other_groups:
            other_group_df = df[df['fl_goes_class'] == other_group]
            other_event_observed = other_group_df[f'event{threshold}'] if f'event{threshold}' in other_group_df.columns else other_group_df['event100']
            other_time = other_group_df['time']
            result = logrank_test(time, other_time, event_observed, other_event_observed, alpha=0.95)
            # Add threshold and group names to the result
            result.threshold = threshold
            result.group_1 = group
            result.group_2 = other_group

            results.append(result)

    # Perform multivariate log-rank test
    multivariate_result = multivariate_logrank_test(df['time'], df['fl_goes_class'], df[f'event{threshold}'])

    return survival_dict, results, multivariate_result


def plot_survival_curves(df, thresholds):

    kmf = KaplanMeierFitter()
    groups = df['fl_goes_class'].unique()
    survival_dict = {}

    for threshold in thresholds:

        for group in groups:
            group_df = df[df['fl_goes_class'] == group]
            event_observed = group_df[f'event{threshold}'] if f'event{threshold}' in group_df.columns else group_df['event100']
            time = group_df['time']
            kmf.fit(time, event_observed=event_observed, label=group)
            survival_dict[group] = kmf.survival_function_
            km_probabilities = survival_dict[group].values

            km_probabilities_1d = [item for sublist in km_probabilities for item in sublist]
            km_probabilities_1d = [round(x, 4) for x in km_probabilities_1d]

            # Convert survival function to a DataFrame
            result_table = {
                'time': survival_dict[group].index,
                'event': kmf.event_table['observed'].values,
                'censored': kmf.event_table['censored'].values,
                'risk set': kmf.event_table['at_risk'].values,
                'KM Survival Probability': km_probabilities_1d
            }
            result_df = pd.DataFrame(result_table)

            result_df.to_csv(f"3_km_estimation/life_tables/data/survival_function_{group}_threshold_{threshold}.csv", index=False)

            # Save the survival DataFrame as a PNG table
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axis('tight')
            ax.axis('off')

            the_table = ax.table(cellText=result_df.values, colLabels=result_df.columns, cellLoc='center', loc='center')

            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(2, 2)

            ax.set_aspect('auto')
            pdf_filename = f"3_km_estimation/life_tables/tables/survival_function_{group}_threshold_{threshold}.pdf"
            pdf_pages = PdfPages(pdf_filename)
            pdf_pages.savefig(fig, bbox_inches='tight')
            pdf_pages.close()
            plt.close(fig)

        # Plot the survival curves
        plt.figure(figsize=(8, 6))

        class_color = {'C': '#4F993C', 'M': '#3B75AF', 'X': '#EA8335'}
        for group, survival_function in survival_dict.items():
            plt.plot(survival_function.index, survival_function[group], label=group, color=class_color[group])

        plt.title(f'Survival Curves (> {threshold} threshold)')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Survival Probability')
        plt.legend()
        # Add the test statistic and p-value to the bottom of the plot
        multivariate_result = multivariate_logrank_test(df['time'], df['fl_goes_class'], df[f'event{threshold}'])
        test_statistic = round(multivariate_result.test_statistic, 4)
        p_value = round(multivariate_result.p_value, 4)
        plt.text(0.5, -0.2, f"Chi-Square Test Statistic: {test_statistic} | P-value: {p_value}", ha='center', transform=plt.gca().transAxes)

        # Adjust the plot layout to create extra space for the text at the bottom
        plt.subplots_adjust(bottom=0.2)

        plt.savefig(f'3_km_estimation/curves/survival_curves_{threshold}.png')
        plt.close()

        results_data = {
            'Threshold': [threshold],
            'Chi-Square Test Statistic': [test_statistic],
            'Degrees of Freedom': [2],
            'P-value': [p_value]
        }
        results_df = pd.DataFrame(results_data)
        csv_filename = f"3_km_estimation/multivariate_log_rank/data/multivariate_logrank_test_{threshold}.csv"
        results_df.to_csv(csv_filename, index=False)

        # Save the results as a PNG
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
        ax.set_title(f"Multivariate Log-Rank (Chi-Square) Test Results for Threshold > {threshold}")
        ax.table(cellText=results_df.values,
                 colLabels=results_df.columns,
                 cellLoc='center', loc='center')
        # Save the plot as a PNG
        png_filename = f"3_km_estimation/multivariate_log_rank/tables/multivariate_logrank_test_{threshold}.png"
        plt.savefig(png_filename, bbox_inches='tight', dpi=300)
        plt.close(fig)


############################################# COX PROPORTIONAL HAZARDS ########################################################

def fit_coxph_model(sse_df, threshold):
    threshold_df = sse_df[['time', 'fl_goes_class', f'event{threshold}']]
    threshold_df_copy = threshold_df.copy()
    threshold_df_copy['fl_goes_class'] = threshold_df_copy['fl_goes_class'].astype('category')
    threshold_df_copy = pd.get_dummies(threshold_df_copy, columns=['fl_goes_class'], drop_first=True)
    cph = CoxPHFitter()
    cph.fit(threshold_df_copy, duration_col='time', event_col=f'event{threshold}')
    schoenfeld_test = proportional_hazard_test(cph, threshold_df_copy, time_transform='rank')

    return cph.summary, schoenfeld_test.summary


def proportional_assumption(sse_df, threshold):
    threshold_df = sse_df[sse_df[f'event{threshold}'] == 1][['time', 'fl_goes_class', f'event{threshold}']]
    threshold_df_copy = threshold_df.copy()
    # Ensure 'fl_goes_class' column contains numeric values (0, 1, 2)
    mapping = {'C': 0, 'M': 1, 'X': 2}
    threshold_df_copy['fl_goes_class'] = threshold_df_copy['fl_goes_class'].map(mapping)

    # Reset the index of threshold_60_df
    threshold_df_copy.reset_index(drop=True, inplace=True)
    cph_summary, schoenfeld_test_summary = fit_coxph_model(sse_df, threshold)

    # Now fit the Cox model
    cph = CoxPHFitter()
    cph.fit(threshold_df_copy, duration_col='time', event_col=f'event{threshold}')

    # Calculate the Schoenfeld residuals using the original DataFrame
    schoenfeld_residuals = cph.compute_residuals(threshold_df_copy, kind='schoenfeld')

    # Create a figure with two axes
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot the Schoenfeld residuals on the first axis
    for idx, covariate in enumerate(schoenfeld_residuals.columns):
        ax1.scatter(x=threshold_df_copy['time'], y=schoenfeld_residuals[covariate], label=covariate)

    ax1.axhline(y=0, color='gray', linestyle='--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Schoenfeld Residuals')
    ax1.legend()
    ax1.set_title('Schoenfeld Residuals Plot for >' + f'event{threshold}' + ' threshold')

    # Extract required columns from Cox PH Summary
    cox_summary = cph_summary[['coef', 'exp(coef)', 'se(coef)']].round(3)
    cox_summary_str = tabulate(cox_summary, headers='keys', tablefmt='plain')
    ax2.axis('off')
    ax2.text(0.05, 0.5, 'Cox PH Summary for >' + f'event{threshold}' + ' threshold:\n\n' + cox_summary_str, fontsize=8, fontfamily='monospace')

    # Extract required columns from Schoenfeld Test Summary
    schoenfeld_summary = schoenfeld_test_summary[['test_statistic', 'p']].round(3)
    schoenfeld_summary_str = tabulate(schoenfeld_summary, headers='keys', tablefmt='plain')
    ax2_2 = ax2.inset_axes([0.5, 0, 0.5, 1], sharey=ax2, sharex=ax2)
    ax2_2.axis('off')
    ax2_2.text(0.05, 0.5, 'Schoenfeld Test Summary for >' + f'event{threshold}' + ' threshold:\n\n' + schoenfeld_summary_str, fontsize=8, fontfamily='monospace')

    plt.tight_layout()
    # Save the plot to a file (e.g., PNG or PDF)
    plt.savefig('4_cox_proportional/graphs/schoenfeld_plot_test_coxph_' + f'{threshold}' + '.png')

    # Convert cox_summary to a dataframe
    cox_summary_df = pd.DataFrame(cox_summary)
    cox_summary_df.columns = ['Coefficient', 'Exp(Coefficient)', 'SE(Coefficient)']
    cox_summary_df = cox_summary_df.round(3)

    # Save cox_summary_df to a CSV file
    cox_csv_filename = f"4_cox_proportional/data/cox_summary_{threshold}.csv"
    cox_summary_df.to_csv(cox_csv_filename, index=False)

    # Convert schoenfeld_summary to a dataframe
    schoenfeld_summary_df = pd.DataFrame(schoenfeld_summary)
    schoenfeld_summary_df.columns = ['Test Statistic', 'p-value']
    schoenfeld_summary_df = schoenfeld_summary_df.round(3)

    # Save schoenfeld_summary_df to a CSV file
    schoenfeld_csv_filename = f"4_cox_proportional/data/schoenfeld_summary_{threshold}.csv"
    schoenfeld_summary_df.to_csv(schoenfeld_csv_filename, index=False)


################################################# PARAMETRIC MODELS ########################################################

def fit_parametric_models(sse_df, thresholds, parametric_models):
    # Dictionary to store AIC and BIC for each model and threshold
    results = {}

    # Loop through each threshold and each solar flare classification to fit the models
    for threshold in thresholds:
        flare_classes = ['C', 'M', 'X']
        for flare_class in flare_classes:
            # Create the DataFrame for the specific threshold and flare class
            threshold_df = sse_df[(sse_df[f'event{threshold}'] == 1) & (sse_df['fl_goes_class'] == flare_class)][['time', 'fl_goes_class', f'event{threshold}']]
            threshold_df_copy = threshold_df.copy()
            # Ensure 'fl_goes_class' column contains numeric values (0, 1, 2)
            mapping = {'C': 0, 'M': 1, 'X': 2}
            threshold_df_copy['fl_goes_class'] = threshold_df_copy['fl_goes_class'].map(mapping)

            # Initialize lists to store AIC and BIC for each model
            aic_list = []
            bic_list = []

            # Fit each parametric model
            for model_name, parametric_model in parametric_models.items():
                parametric_model.fit(threshold_df_copy['time'], threshold_df_copy[f'event{threshold}'])

                # Append AIC and BIC to the lists
                aic_list.append(np.round(parametric_model.AIC_, 3))
                bic_list.append(np.round(parametric_model.BIC_, 3))

            # Store AIC and BIC for the specific threshold and flare class in the results dictionary
            results[(threshold, flare_class)] = {'AIC': aic_list, 'BIC': bic_list}

    return results


def get_equation_for_best_model(model):
    if isinstance(model, ExponentialFitter):
        coef_1 = model.params_[0]  # Assuming the coefficient is the first parameter
        inv_coef = 1 / coef_1
        exp_term = -inv_coef
        equation = r"$S(t) = e^{" + f"{exp_term:.8f}" + r" t}$"
        return equation, [coef_1, exp_term]
    elif isinstance(model, WeibullFitter):
        lambda_ = model.lambda_
        rho = model.rho_
        equation_weibull = rf"$S(t) = \exp \left( - \left( \dfrac{{t}}{{{lambda_:.6f}}} \right)^{{{rho:.6f}}} \right)$"
        return equation_weibull, [lambda_, rho]
    elif isinstance(model, LogNormalFitter):
        mu = model.mu_
        sigma = model.sigma_
        equation = rf"$S(t) = 1 - \Phi \left(\dfrac{{\ln(t) - {mu:.6f}}} {{{sigma:.6f}}}\right)$"
        return equation, [mu, sigma]
    elif isinstance(model, LogLogisticFitter):
        alpha = model.alpha_
        beta = model.beta_
        equation_loglogistic = fr"$S(t) = \frac{{1}}{{1 + \left( \frac{{t}}{{{alpha:.6f}}} \right)^{{{beta:.6f}}}}}$"
        return equation_loglogistic, [alpha, beta]
    else:
        return "Unknown Model"


def create_parametric_model_table(sse_df, thresholds, parametric_models):
    # Create the tables
    results = fit_parametric_models(sse_df, thresholds, parametric_models)
    best_models = {}

    for threshold in thresholds:
        headers = ['Criterion'] + list(parametric_models.keys())

        for classification in ['C', 'M', 'X']:
            table_data = [headers]

            aic_values = [f"{value:.3f}" for value in results[(threshold, classification)]['AIC']]

            table_data.append(['AIC'] + aic_values)

            # Find the index of the minimum AIC and BIC values for the current classification
            min_aic_index = min(range(len(aic_values)), key=aic_values.__getitem__)
            # Get the names of the best models based on AIC and BIC
            best_model_aic = list(parametric_models.keys())[min_aic_index]
            best_models[classification] = best_model_aic

            # Fit the best model using the current classification's data
            # Filter data by classification and threshold
            filtered_df = sse_df[(sse_df['fl_goes_class'] == classification) & (sse_df[f'event{threshold}'])]
            # print(filtered_df)
            threshold_df = filtered_df[['time', f'event{threshold}']]
            best_model_1 = parametric_models[best_model_aic]
            best_model_1.fit(threshold_df['time'], threshold_df[f'event{threshold}'])

            # Extract and store the equation for the best model
            equation_1, coef_1 = get_equation_for_best_model(best_model_1)

            time_points = list(set(threshold_df['time']))
            time_points = np.insert(time_points, 0, [0.0]).astype(int)
            time_points.sort()  # Sort the list in ascending order if needed
            time_points = np.array(time_points)

            survival_probabilities = 0
            # Calculate survival probabilities using the equation for each time point
            if best_model_aic == 'exponential':
                survival_probabilities = np.exp(coef_1[1] * time_points)
            elif best_model_aic == 'weibull':
                survival_probabilities = np.exp(-(time_points / coef_1[0]) ** coef_1[1])
            elif best_model_aic == 'loglogistic':
                survival_probabilities = 1 / (1 + (time_points / coef_1[0]) ** coef_1[1])
            elif best_model_aic == 'lognormal':
                survival_probabilities = (np.log(time_points) - coef_1[0]) / coef_1[1]
                survival_probabilities = 1 - norm.cdf(survival_probabilities)

            # Get the Kaplan-Meier estimator for the current classification
            kmf = KaplanMeierFitter()
            event_observed = threshold_df[f'event{threshold}'] if f'event{threshold}' in threshold_df.columns else threshold_df['event100']
            time = threshold_df['time']
            kmf.fit(time, event_observed=event_observed, label=classification)
            kmf_estimator = kmf.survival_function_[classification]
            kmf_event_table = kmf.event_table

            # Compare survival probabilities and create a comparison column
            comparison = [round(prob_km, 2) == round(prob_parametric, 2) for prob_km, prob_parametric in zip(kmf_estimator.values, survival_probabilities)]

            # Plot the parametric curve
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, survival_probabilities, label="Parametric Curve")
            # Plot the Kaplan-Meier step curve
            plt.step(kmf_estimator.index, kmf_estimator.values, where="post", label="Kaplan-Meier Curve")

            # Plot the matching points where comparison is True
            matching_points = [t for t, is_matching in zip(kmf_estimator.index, comparison) if is_matching]
            matching_probs = [prob_km for prob_km, is_matching in zip(kmf_estimator.values, comparison) if is_matching]
            plt.scatter(matching_points, matching_probs, color='red', marker='o', s=50, label="Matching Points")

            title = f"Threshold >{threshold} for {classification} : {best_model_aic}"
            plt.text(0.5, -0.20, title, ha='center', va='center', transform=plt.gca().transAxes)
            table_str = tabulate(table_data, headers='firstrow', tablefmt='plain')
            plt.text(0.5, -0.29, table_str, ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.8, 0.7, f"{equation_1}", ha='center', va='center', transform=plt.gca().transAxes)

            # Add labels and legend
            plt.xlabel("Time")
            plt.ylabel("Survival Probability")
            plt.title(f"Parametric Survival Curve for Classification {classification}, Threshold {threshold}")
            plt.legend([f'Parametric Curve ({best_model_aic})', 'Kaplan-Meier Curve', 'Matching Points'])
            plt.subplots_adjust(bottom=0.25)
            plt.savefig(f"5_parametric_analysis/curves/parametric_survival_curve_{classification}_{threshold}" + '.png')
            plt.close()

            # Build the desired table
            result_table = pd.DataFrame({
                'time': kmf_estimator.index,
                'risk set': kmf_event_table['at_risk'],
                'KM Survival Probability': kmf_estimator.values.round(4),
                'Parametric Survival Probability': survival_probabilities.round(4),
                'Comparison': comparison
            })

            csv_filename = f"5_parametric_analysis/data/km_parametric_probabilities_{classification}_{threshold}.csv"
            result_table.to_csv(csv_filename, index=False)

            # Create a table plot using Matplotlib
            fig, ax_result = plt.subplots(figsize=(10, 6))

            ax_result.axis('off')
            # Format the numbers to have 4 decimal points
            result_table_formatted = result_table.copy()
            result_table_formatted[['KM Survival Probability', 'Parametric Survival Probability']] = result_table[
                ['KM Survival Probability', 'Parametric Survival Probability']].round(4)
            # Calculate the height of the table based on the number of rows
            ax_result.table(cellText=result_table_formatted.values, colLabels=result_table.columns, cellLoc='center', loc='center')

            # Save the table plot as a PDF
            pdf_filename = f"5_parametric_analysis/life_tables/km_parametric_probabilities_{classification}_{threshold}.pdf"
            pdf_pages = PdfPages(pdf_filename)
            pdf_pages.savefig(fig, bbox_inches='tight')
            pdf_pages.close()
            plt.close(fig)

    return best_models
