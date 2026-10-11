# Studying-Chytrid-Fungus-Using-Supervised-Machine-Learning

## Project Topic

Chytrid fungus is a skin disease that eventually causes death in many species, and contributes to the decline in many amphibian species all around the globe. I wanted to create a few classification models to try and predict whether an amphibian does or does not test positive for chytrid fungus, based on data about the animal, and its surrounding environment. I also want to make a separate model with just environmental data (nothing about the specific animals) to see if just that data is enough to predict whether a animal may test positive for chytrid fungus. 

## Data

This project uses a dataset from the U.S. Geological Survey that shows the presence of absence of of chytrid fungus all around North America (mostly the U.S.). This survey occured from May 2014 to August 2017, and each amphibian was swabbed to see if they tested positive for chytrid fungus or not. The dataset includes information about the animal itself, as well as the surrounding environment.

https://data.usgs.gov/datacatalog/data/USGS:5af5ef0de4b0da30c1b5f9eb

The dataset is a 7.58 MB csv file, and includes 11,189 samples and 70 features. I have included each feature and a brief definition below

* SwabID: Barcode or submitter-assigned identification number. Categorical.

* CollectionDate: Date of field swab collection. Numeric.

* TaxaID: Numeric code for species corresponding to the Integrated Taxonomic Information System (ITIS) found at https://www.itis.gov/. Categorical.

* ScientificName: Species Name. Categorical.

* SiteName: Submitter-assigned site name where swabs were collected. Categorical.

* State: Categorical.

* County: Categorical.

* Latitude: Numeric.

* Longitude: Numeric.

* Elevation_m: Numeric.

* Sex: Can be Male, Female, or Unknown. Categorical.

* Age: Written as NA if age was not ascertained. Otherwise Adult, Juvenile, or Larval. Categorical.

* BodyLength: The body length of the swabbed individual. For amphibians, this is the straight-line distance from the tip of the snout to the vent. It does not include the tail length. An NA is denoted for those records where body length was not measured. Numeric.

* CaptureEnvironment: The habitat in which the swabbed animal was captured. Either aquatic or terrestrial. Categorical.

* Bsal.Result: binary result for Bsal PCR, positive = detected, negative = not detected. Bsal is salamander chytrid fungus, which was not detected. This is separate from the result for amphibian chytrid fungus, which was detected. Categorical.

* Case: Internal lab identifier for group of samples received in the same shipment. Categorical.

* Acc: Internal lab identifier for individual sample within a Case. Categorical.

* Bdct: Cycle threshold value from PCR reaction (max=40). Lower values indicate more target DNA in sample. Measured at the PCR cycle where a signal was detected at 5% of max fluorescence and subsequent signal followed a true amplification curve. NA = no signal detected; No Amp = signal detected, but due to signal noise and not amplification of the target; equivocal = signal detected during last cycle with likely amplification, but reaction cycles stopped before true amplification could be determined. Numeric.

* Bd.Result: Interpretation of quantitative PCR results. Positive = true amplification signal detected; Negative = no fluorescent signal detected; No Amp = signal detected, but due to signal noise and not amplification of the target. Manually confirmed during lab QA/QC; equivocal = signal detected during last cycle with likely amplification, but reaction cycles stopped before true amplification could be determined. Categorical.

* copyNum: Calculated estimate of the number of target DNA copies per sample. Numeric.

* Gesp: Genus and species used to estimate phylogenetic distance. Categorical.

* order: taxonomic order. Categorical.

* siteID: numeric index that groups samples within 0.2km to a site. Categorical.

* siteVID: numeric index for site visits that occurred greater than 5 days apart at each site. Categorical.

* reg: NOAA climate region where the site was sampled. Categorical.

* KGzone: 3rd level Köppen-Geiger climate zone where the sample was collected. Categorical.

* KG5: 2nd level Köppen-Geiger climate zone where the sample was collected. Categorical.

* KG3: 1st level Köppen-Geiger climate zone where the sample was collected. Categorical.

* m1max: maximum daily temperature estimated at the day of sampling. Numeric. 

* m7max, m10max, m20max, m30max, m40max, m50max, m60max, m70max, m80max, m90max, m100max, m110max, m120max: average maximum daily temperature estimated for the 7, 10, 20, etc days preceding sampling. Numeric.

* m1min: minimum daily temperature estimated at the day of sampling. Numeric.

* m7min, m10min, m20min, m30min, m40min, m50min, m60min, m70min, m80min, m90min, m100min, m110min, m120min: average minimum daily temperature estimated for the 7, 10, 20, etc days preceding sampling. Numeric.

* t1precip: total precipitation estimated on the day of sampling. Numeric.

* t7precip, t10precip, t20precip, t30precip, t40precip, t50precip, t60precip, t70precip, t80precip, t90precip, t100precip, t110precip, t120precip: total precipitation estimated over the 7, 10, 20, etc days preceding sampling. Numeric.

## Data Cleaning and EDA

Does it include clear explanations on how and why a cleaning is performed? (e.g.) the author decided to drop a feature because it had too many NaN values and the data cannot be imputed (e.g.) the author decided to impute certain values in a feature because the number of missing values were small and he/she was able to find similar samples OR, he/she used an average value or interpolated value, etc. (e.g.) the author removed some features because there are too many of them and they are not relevant to the problem, or he/she knows only a few certain features are important based on their domain knowledge judgement. (e.g.) the author removed certain sample (row) or a value because it is an outlier.
Does it include clear explanations on how and why an certain analysis (EDA) is performed?
Does it have proper visualization?
Does it have proper analysis? E.g. histogram, correlation matrix, feature importance (if possible), etc.
Does it have conclusions or discussions? E.g. the EDA summary, findings, discussing foreseen difficulties and/or analysis strategy.


Starting off, I had to drop a few features. Some were more sample identifiers: **SwabID**, **Case**, and **Acc**. Some were essentially the same as other features already in the table, for example each scientific name was assigned a TaxaID, so I didn't need both. Those dropped due to being the same as other features: **TaxaID**, **Gesp**, **siteID**, **siteVID**. Some features were dropped because they were used to determine the values of other features, like a few columns were used to determine the value in the Bd column, which are: **Bdct** and **copyNum**. I chose to drop the **SiteName** and **County** features because I thought the locations were too specific for how broad the study location was. I also dropped **Bsal.Result** because all of the values were negative. 

I wanted to extract data from a couple of the features. I dropped the **ScientificName** feature and replaced it with separate Genus and Species features. I also dropped the **CollectionDate** feature, and replaced it with separate Month and Year features.  

Then I looked at the features with missing values, and dropped all rows containing missing values. This brought me from 11,189 samples to 7,663 samples. 

Then, I wanted to scale some of my features with larger values, so all of my numbers are within a similar range, which is necessary for logistic regression. I changed Year, Elevation, and all of the precipitation features. 

The next thing I had to do to prep my dataset was convert all of my categorical variables to numeric variables, because I can't use categorical variables in my logistic regression model. 

Finally before making my model, I wanted to check for correlations. I first removed all but one temperature and precipitation variable, because I knew that all the precipitation and temperature measurements would be correlated. I kept m60max and t60precip_100. While there were some correlated variables, none seemed particularly surprising to me, like the KGzone is related to KG5 and KG3, so I will keep all the variables for now.

![Correlation Heatmap](images/correlation_heatmap.png)


4. Model Building / Model Choice (15 pts)

Which models are suitable for my problem and why?
Compare with a baseline model(s); baseline means a simple, easy to check model.
Grading Criteria:
  How many models have been used and compared? You may give partial credit based on your judgement.
Compares 1-2 models (5 pts)
Compares 3 or more models (10 pts)
Compares multiple models (3 or more) and includes reasoning of why chosen models are suitable for the problem (before getting results). Also may include a simple (or lesser performing model) for baseline comparison. (15 pts)
5. Model Training (15 pts)

How to find best hyperparameters of a model?
Is there overfitting? How to check, how to mitigate?
Are all features needed? Why or why not? If so, how can I choose features?
What are the important features?
Do I have imbalanced classes in my data? What to do?
Grading Criteria:
Base 5 pts: Did basic model training with some hyperparameter tuning.
Add any additional points. You can also give partial points to these based on your judgement.
+ 3 pts: Did a though hyperparameter tuning (manual or uses sklearn library's grid search function).
+ 3 pts: Include train and test error analysis and visualization (e.g. elbow curve) while searching hyper parameters.
+ 3 pts: Use of cross validation or methods to monitor or mitigate overfitting
+ 3 pts: Use feature selection or feature engineering. Or does extra analysis on feature rankings (feature importance)
+ 3 pts: If the data's classes are imbalanced - uses methods to mitigate class imbalance (e.g.) oversampling or undersampling or using synthesized data (SMOTE) or setting model's option to adjust class weights if available. Must explain why they made these choices explicitly. Using a model library's default options (even when it has set to use these weights by default) will not qualify.
6. Results and Analysis (20 pts)

Things to consider:
Does it have a summary of results and analysis?
Does it have a proper visualization? (e.g. tables, graphs/plots, heat maps, statistics summary with interpretation etc)
Does it use different kinds of evaluation metric properly? (e.g. if your data is imbalance, there are other metrics F1, ROC or AUC better than mere accuracy). Also does it explain why they chose the metric?
Does it iterate the training and evaluation process and improve the performance? Does it address selecting features through the iteration process?
Did the author compare the results from the multiple models and carry out appropriate comparison?
Grading Criteria:
Example Score Breakdown: Showing basic result +10 pts with a good amount of visualization +5 pts, tries different evaluation metrics +5 pts, iterates training/evaluating and improves performance +5, shows/discusses model comparison +5 pts.
7. Discussion and Conclusion (10 pts)

Provide conclusions from your result. Discuss what went well or not well. Discuss any suggestions to improve.
Grading Criteria:
Example Score Breakdown: Basic reiteration of results +5 pts, discusses why something didn't work +5 pts (additional +5 pts for going back to modeling and trying more analysis), suggests ways to improve +5 pts.