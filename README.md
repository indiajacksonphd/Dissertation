# Implementation of Statistical Machine Learning Models for Space Weather Prediction in a Cloud Computing Environment

There are 3 parts to this project: <br>

- Use of survival analysis for time to detection of solar energetic particles (SEPs)<br>
- Use of random survival trees for time to detection of SEPs <br>
- Analysis for time to detection of SEPs conducted remotely in an AWS cloud infrastructure <br>


# Solar Survival Package

The solarSurvival package is a comprehensive and innovative toolkit designed to revolutionize the analysis, prediction, and understanding of SEPs in the realm of space weather research. Solar flares and their associated energetic particles hold critical implications for technology, astronaut health, and space missions. This package goes beyond traditional methods, offering a comprehensive approach that encompasses advanced statistical survival analysis, cloud computing, and will be updated with integration of cutting-edge machine learning techniques.

At its core, solarSurvival facilitates intricate survival analysis by leveraging a combination of non-parametric, semi-parametric, and parametric models including the Kaplan-Meier (KM) Estimator and Cox Proportional Hazards (Cox PH) Models alongside the exponential, Weibull, lognormal, and loglogistic distributions. These techniques empower users to explore censored data, calculate event probabilities over time, and uncover hidden patterns in SEP occurrences. The package harnesses the power of cloud-based resources, ensuring efficient data preprocessing, analysis, and visualization. By remotely processing data using Amazon Web Services (AWS) cloud computing, researchers can optimize their analysis, setting the stage for more extensive big data and machine learning endeavors.

The solarSurvival package will be upgraded with the integration of random survival trees, a state-of-the-art machine learning technique. After deriving insights from parametric survival equations, the package will synthesize data and employ random survival trees to predict SEP events. By combining parametric survival analysis with machine learning, the package will offer a unique blend of interpretability and predictive accuracy. Random survival trees enable the identification of complex relationships within SEP data, facilitating the discovery of influential factors and enhancing the reliability of event predictions.

The solarSurvival package is a dedicated showcase of my in-depth research and analysis into SEPs, designed to present my findings and capabilities in the realm of space weather research. The primary intent of the solarSurvival package is to serve as a visual representation of my research process and outcomes. While the code is made available for exploration and manipulation, it's important to note that the package isn't designed for generic reusability. Instead, it's tailored to present my specific research results and to provide a platform for replication and validation of my findings. By offering an interactive and engaging experience, it allows viewers to gain a deeper understanding of my research methodology, findings, and the potential impact of SEP events.

## Versioning and Updates

The solarSurvival package follows semantic versioning (SemVer). This means that version numbers are assigned based on the significance of changes introduced in each release. A version number consists of three parts: MAJOR.MINOR.PATCH.

- **MAJOR:** Incremented for backward-incompatible changes or major new features.
- **MINOR:** Incremented for backward-compatible feature additions or enhancements.
- **PATCH:** Incremented for backward-compatible bug fixes or small improvements.

#### Version 1.0.0 (2023-08-20)

Initial Release:
- Comprehensive survival analysis using on solar flare classification (C, M, and X) as a predictor
- Non-parametric, semi-parametric, and parametric analysis including the Kaplan-Meier Estimator, Cox Proportional Hazards Models, Schoenfeld Residuals, and exponential, Weibull, lognormal, and loglogistic fitting. <br>
- Cloud-based data preprocessing, analysis, and visualization using AWS. <br>
- Welcome message with project overview and results directory information. <br>

#### Version 1.1.0 (Upcoming)

Upcoming Update:
- Integration of random survival trees for predictive modeling using solar flare classification and flare location as predictors.
- A comparison of original vs synthetic data based on statistical parametric models.
- Additional examples and usage scenarios.
- Improved documentation and code optimizations.

#### Project Structure and Folder Descriptions

- **datasets**: Contains the survival analysis dataset extracted from the GSEP catalog.
- **1_exploratory_analysis_all_events**: Contains data summaries, frequency tables, histograms, and box plots for all events.
- **2_exploratory_analysis_classification**: Contains data summaries, pie charts, and box plots based on solar flare class.
- **3_km_estimation**: Contains survival "step" curves and survival probabilities by threshold based on solar flare class.
- **4_cox_proportional**: Compares and analyzes significance of solar flare classification using Cox PH Models and Schoenfeld Tests.
- **5_parametric_analysis**: Contains parametric equations, curves, and survival probabilities of parametric models along with their associated survival "step" curves.


### Keeping Up-to-Date

For raw code access and to stay updated with the latest changes and enhancements, you can regularly check the [GitHub repository](https://github.com/indiajacksonphd/Dissertation/).


## Requirements

The Solar Survival Package relies on the following libraries and dependencies. They will be automatically installed when you install the package:

- pandas
- boto3
- numpy
- seaborn
- matplotlib
- scipy
- tabulate
- lifelines


## Installation and Usage

Before you begin, ensure that you have the following:

- Python 3.x installed on your system.
- A working internet connection to download required packages.

### Integrated Development Environment (IDE):

1. Open your favorite IDE.

2. Depending on your environment, you may need to install the required packages before installing solarSurvival. Run the following commands to install the necessary dependencies:


    ```bash
    pip3 install pandas boto3 numpy seaborn matplotlib scipy tabulate lifelines
    ```

3. Install the solarSurvival package using the following command:

    ```bash
    pip3 install -i https://test.pypi.org/simple/ solarSurvival
    ```

4. Create a new Python script (e.g., main.py) in your project directory.

5. Copy and paste the following code into your Python script:

    ```bash
    from solar_Survival.solarSurvival import welcome_message, data_extraction, results

    def main():
        welcome_message()
        data_extraction()
        results()

    if __name__ == '__main__':
        main()

    ```

6. Run your Python script. The script will execute the welcome_message, data_extraction, and results functions from the solarSurvival package.

### Terminal:

1. Open your terminal and make sure that you are in the directory where you want to work.
2. Run the following command to find the location of the package and save it into an environmental variable:

    ```bash
    export SOLAR_SURVIVAL_PATH=$(pip3 show solarSurvival | grep -E '^Location:' | awk '{print $2}')
    ```

3. Run the following command to run the package:

    ```bash
    python3 $SOLAR_SURVIVAL_PATH/solar_Survival/solarSurvival.py
    ```

### Command Prompt (Windows):

1. Open your command prompt and make sure that you are in the directory where you want to work.
2. Run the following command to find the location of the package and save it into an environmental variable:

    ```batch
    for /f "tokens=2 delims= " %A in ('pip show solarSurvival ^| find "Location"') do set SOLAR_SURVIVAL_PATH=%A
    ```

3. Run the following command to run the package:

    ```batch
    python %SOLAR_SURVIVAL_PATH%\solar_Survival\solarSurvival.py
    ```

### PowerShell (Windows):

1. Open your PowerShell and make sure that you are in the directory where you want to work.
2. Run the following command to find the location of the package and save it into an environmental variable:

    ```powershell
    $SOLAR_SURVIVAL_PATH = (pip show solarSurvival | Select-String -Pattern '^Location:' | ForEach-Object { $_.ToString().Split(':')[1].Trim() })
    ```

3. Run the following command to run the package:

    ```powershell
    python $SOLAR_SURVIVAL_PATH\solar_Survival\solarSurvival.py
    ```

You will see a welcome message along with information about where you can find the results.


## License
This package is provided under the [MIT License](https://github.com/your-username/your-repo/blob/main/LICENSE).


## Contact
For any questions or inquiries, please contact India Jackson at ijackson1@gsu.edu.

