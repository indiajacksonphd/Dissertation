# Dissertation project: Implementation of Statistical Machine Learning Models for Space Weather Predictin in a Cloud Computing Environment

There are 3 parts to this project:
	1) Use of survival analysis for time to detection of solar energetic particles
	2) use of random survival trees for time to detection of solar energetic particles
	3) analysis for time to detection of solar energetic particles conducted 100% remotely in an AWS cloud infrastructure


# Solar Survival Package

The "solarSurvival" package is a comprehensive and innovative toolkit designed to revolutionize the analysis, prediction, and understanding of solar energetic particle events (SEPs) in the realm of space weather research. Solar flares and their associated energetic particles hold critical implications for technology, astronaut well-being, and space missions. This package goes beyond traditional methods, offering a holistic approach that encompasses advanced statistical survival analysis, cloud computing, and will be updated with integration of cutting-edge machine learning techniques.

At its core, "solarSurvival" facilitates intricate survival analysis by leveraging a combination of non-parametric, semi-parametric, and parametric models including the Kaplan-Meier (KM) Estimator and Cox Proportional Hazards (Cox PH) Models alongside the exponential, Weibull, log-normal, and loglogistic distributions. These techniques empower users to explore censored data, calculate event probabilities over time, and uncover hidden patterns in SEP occurrences. The package seamlessly harnesses the power of cloud-based resources, ensuring efficient data preprocessing, analysis, and visualization. By remotely processing data using Amazon Web Services (AWS) cloud computing, researchers can optimize their analysis, setting the stage for more extensive big data and machine learning endeavors.

The "solarSurvival" package will be upgraded with the integration of random survival trees, a state-of-the-art machine learning technique. After deriving insights from parametric survival equations, the package will allow users to synthesize data and employ random survival trees to predict SEP events. By combining parametric survival analysis with machine learning, the package will offer a unique blend of interpretability and predictive accuracy. Random survival trees enable the identification of complex relationships within SEP data, facilitating the discovery of influential factors and enhancing the reliability of event predictions.

Upon installation, the package provides an informative welcome message that sets the stage for users, offering a clear project overview and directions to access generated files and results. Researchers, space weather enthusiasts, and data scientists alike can now utilize the "solarSurvival" package to gain deeper insights into the dynamics of SEP events, predict occurrences with greater accuracy, and contribute to the advancement of space weather prediction and mitigation strategies. With its seamless integration of statistical analysis, cloud computing, and machine learning, the "solarSurvival" package empowers users to navigate the complexities of solar energetic particle events and enhance the resilience of technology and astronaut health in the face of these challenges.

## Versioning and Updates

The "solarSurvival" package follows semantic versioning (SemVer). This means that version numbers are assigned based on the significance of changes introduced in each release. A version number consists of three parts: MAJOR.MINOR.PATCH.

- **MAJOR:** Incremented for backward-incompatible changes or major new features.
- **MINOR:** Incremented for backward-compatible feature additions or enhancements.
- **PATCH:** Incremented for backward-compatible bug fixes or small improvements.

### Changelog

#### Version 1.0.0 (2023-08-20)

Initial Release:
- Comprehensive survival analysis methods including Kaplan-Meier Estimator and Cox Proportional Hazards Models.
- Cloud-based data preprocessing, analysis, and visualization using AWS.
- Welcome message with project overview and results directory information.

#### Version 1.1.0 (Upcoming)

Upcoming Update:
- Integration of random survival trees for predictive modeling.
- Additional examples and usage scenarios.
- Improved documentation and code optimizations.

### Keeping Up-to-Date

To stay updated with the latest changes and enhancements, you can regularly check the [GitHub repository](https://github.com/indiajacksonphd/Dissertation/) and monitor the [Changelog](https://github.com/indiajacksonphd/Dissertation/blob/main/CHANGELOG.md) for detailed information about each version's release notes.

We value user feedback and are committed to continually improving the "solarSurvival" package. If you have suggestions, bug reports, or feature requests, please feel free to [open an issue](https://github.com/indiajacksonphd/Dissertation/issues) on GitHub. Your contributions are highly appreciated in making this package even better!

Remember to keep your changelog (if you create one) and the information in this section updated as you release new versions of your package.


## Requirements
The Solar Survival Package relies on the following libraries and dependencies. They will be automatically installed when you install the package:

pandas==1.3.3
boto3==1.18.120
numpy==1.21.2
seaborn==0.11.2
matplotlib==3.4.3
scipy==1.7.1
tabulate==0.8.9
lifelines==0.28.2


## Installation and Usage

1. Open your terminal/command prompt.
2. Copy and paste the following command to install the "solarSurvival" package:
	pip install solarSurvival
3. After installation, you can run the package by entering the following command:
	python3 solarSurvival.py

You will see a welcome message along with information about where you can find the results.


## License
This package is provided under the MIT License.

## Contact
For any questions or inquiries, please contact India Jackson at ijackson1@gsu.edu.

