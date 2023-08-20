# Implementation of Statistical Machine Learning Models for Space Weather Predictin in a Cloud Computing Environment

There are 3 parts to this project: <br>
	1) Use of survival analysis for time to detection of solar energetic particles (SEPs)<br>
	2) Use of random survival trees for time to detection of SEPs <br>
	3) Analysis for time to detection of SEPs conducted remotely in an AWS cloud infrastructure <br>


## Solar Survival Package

The "solarSurvival" package is a comprehensive and innovative toolkit designed to revolutionize the analysis, prediction, and understanding of SEPs in the realm of space weather research. Solar flares and their associated energetic particles hold critical implications for technology, astronaut health, and space missions. This package goes beyond traditional methods, offering a comprehensive approach that encompasses advanced statistical survival analysis, cloud computing, and will be updated with integration of cutting-edge machine learning techniques.

At its core, "solarSurvival" facilitates intricate survival analysis by leveraging a combination of non-parametric, semi-parametric, and parametric models including the Kaplan-Meier (KM) Estimator and Cox Proportional Hazards (Cox PH) Models alongside the exponential, Weibull, log-normal, and loglogistic distributions. These techniques empower users to explore censored data, calculate event probabilities over time, and uncover hidden patterns in SEP occurrences. The package harnesses the power of cloud-based resources, ensuring efficient data preprocessing, analysis, and visualization. By remotely processing data using Amazon Web Services (AWS) cloud computing, researchers can optimize their analysis, setting the stage for more extensive big data and machine learning endeavors.

The "solarSurvival" package will be upgraded with the integration of random survival trees, a state-of-the-art machine learning technique. After deriving insights from parametric survival equations, the package will synthesize data and employ random survival trees to predict SEP events. By combining parametric survival analysis with machine learning, the package will offer a unique blend of interpretability and predictive accuracy. Random survival trees enable the identification of complex relationships within SEP data, facilitating the discovery of influential factors and enhancing the reliability of event predictions.

The "solarSurvival" package is a dedicated showcase of my in-depth research and analysis into SEPs, designed to present my findings and capabilities in the realm of space weather research. The primary intent of the "solarSurvival" package is to serve as a visual representation of my research process and outcomes. While the code is made available for exploration and manipulation, it's important to note that the package isn't designed for generic reusability. Instead, it's tailored to present my specific research results and to provide a platform for replication and validation of my findings. By offering an interactive and engaging experience, it allows viewers to gain a deeper understanding of my research methodology, findings, and the potential impact of SEP events.

## Versioning and Updates

The "solarSurvival" package follows semantic versioning (SemVer). This means that version numbers are assigned based on the significance of changes introduced in each release. A version number consists of three parts: MAJOR.MINOR.PATCH.

- **MAJOR:** Incremented for backward-incompatible changes or major new features. <br>
- **MINOR:** Incremented for backward-compatible feature additions or enhancements. <br>
- **PATCH:** Incremented for backward-compatible bug fixes or small improvements. <br>

### Changelog

#### Version 1.0.0 (2023-08-20)

Initial Release:
- Comprehensive survival analysis using on solar flare classification (C, M, and X) as a predictor
- Non-parametric, semi-parametric, and parametric anamlysis including the Kaplan-Meier Estimator, Cox Proportional Hazards Models, Schoenfeld Residuals, and exponential, Weibull, lognormal, and loglogistic fitting. <br>
- Cloud-based data preprocessing, analysis, and visualization using AWS. <br>
- Welcome message with project overview and results directory information. <br>

#### Version 1.1.0 (Upcoming)

Upcoming Update:
- Integration of random survival trees for predictive modeling using solar flare classification and flare location as predictors.
- A comparison of original vs synthetic data based on statisical parametric models.
- Additional examples and usage scenarios.
- Improved documentation and code optimizations.

### Keeping Up-to-Date

To stay updated with the latest changes and enhancements, you can regularly check the [GitHub repository](https://github.com/indiajacksonphd/Dissertation/).


## Requirements
The Solar Survival Package relies on the following libraries and dependencies. They will be automatically installed when you install the package: <br>

pandas==1.3.3 <br>
boto3==1.18.120 <br>
numpy==1.21.2 <br>
seaborn==0.11.2 <br>
matplotlib==3.4.3 <br>
scipy==1.7.1 <br>
tabulate==0.8.9 <br>
lifelines==0.28.2 <br>


## Installation and Usage

1. Open your terminal/command prompt.
2. Copy and paste the following command to install the "solarSurvival" package: <br>
	pip install solarSurvival
3. After installation, you can run the package by entering the following command: <br>
	python3 solarSurvival.py

You will see a welcome message along with information about where you can find the results.


## License
This package is provided under the MIT License.

## Contact
For any questions or inquiries, please contact India Jackson at ijackson1@gsu.edu.

