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

Starting off, I had to drop a few features. Some were more sample identifiers: **SwabID**, **Case**, and **Acc**. Some were essentially the same as other features already in the table, for example each scientific name was assigned a TaxaID, so I didn't need both. Those dropped due to being the same as other features: **TaxaID**, **Gesp**, **siteID**, **siteVID**. Some features were dropped because they were used to determine the values of other features, like a few columns were used to determine the value in the Bd column, which are: **Bdct** and **copyNum**. I chose to drop the **SiteName** and **County** features because I thought the locations were too specific for how broad the study location was. I also dropped **Bsal.Result** because all of the values were negative. 

I wanted to extract data from a couple of the features. I dropped the **ScientificName** feature and replaced it with separate Genus and Species features. I also dropped the **CollectionDate** feature, and replaced it with separate Month and Year features.  

Then I looked at the features with missing values, and dropped all rows containing missing values. This brought me from 11,189 samples to 7,663 samples. 

Then, I wanted to scale some of my features with larger values, so all of my numbers are within a similar range, which is necessary for logistic regression. I changed Year, Elevation, and all of the precipitation features. 

The next thing I had to do to prep my dataset was convert all of my categorical variables to numeric variables, because I can't use categorical variables in my logistic regression model. 

Finally before making my model, I wanted to check for correlations. I first removed all but one temperature and precipitation variable, because I knew that all the precipitation and temperature measurements would be correlated. I kept m60max and t60precip_100. While there were some correlated variables, none seemed particularly surprising to me, like the KGzone is related to KG5 and KG3. My issue is I don't know which of the weather-related variables are most important, so I can't get rid of any of them right now, but it may cause some issues later on. 

![Correlation Heatmap](images/correlation_heatmap.png)

## Model Building / Model Choice

There are many models suitable for this problem. The first model I wanted to use was logistic regression, because it is a very common to use when there are two categorical outcomes, and it is a good baseline model. I also wanted to try a Random Forest model, because I know I have too many correlated features, and I thought Random Forest could help me prevent overfitting, and determine which of the correlated features are most important. The last model I wanted to try was CatBoost, because this is better for datasets that contain a high number of categorical features, which my dataset does. For CatBoost, I can specify which of my features are categorical versus numeric, which I can't do in logisitc regression or random forest. I had to assign arbitrary numbers to all of the categorical variables in my logistic regression and random forest models, which would imply that there is some order to the variables when there isn't. 

## Model Training / Results

### Logistic Regression

For my logisitc regression model, I wanted to start by checking my hyperparameters. For logistic regression, I just wanted to check the values of C. With C=0.1, the model's accuracy was 0.783, which is slightly better than the original 0.780, so I will use this tuned model going forward.

![Logistic Regression Validation Accuracy](images/LG_GridSearch_Heatmap.png)

Next I used kfold cross validation to test for overfitting. The training average accuracy was 77.5%, and the test average accuracy was 77%, so I did not see signs of overfitting. 

Note: I did not check the feature importance for my logisitic regression model as that is not a feature available for logistic regression models.

As another way to test the performance of the model, I made an ROC curve, which shows that the model performs well.

![ROC Curve Logistic Regression](images/LogReg_AUC_Score.png)

### Random Forest

For my random forest model, I wanted to start by checking my hyperparameters. For random forets, I wanted to check the n_estimators, max_depth, min_samples_split. This predicted the best model to be n_estimators = 300, max_depth = 10, and min_samples_split = 5, and the model accuracy slightly increased from 0.82 to 0.83, so I will use this model going forward

Next I used kfold cross validation to test for overfitting. The training average accuracy was 83%, and the test average accuracy was 81%, which I wouldn't see as a major concern for overfitting.

Next, I was curious how the model ranked feature importance, so I made a list of the top 20 features

![RF top 20 features](images/RF_top_20.png)

As another way to test the performance of the model, I made an ROC curve, which shows that the model performs well.

![ROC Curve Random Forest](images/RF_AUC_Score.png)

### CatBoost

For my CatBoost model, I wanted to start by checking my hyperparameters. For CatBoost, I wanted to check iterations, depth, and learning_rate. This predicted the best model to be iterations = 1000 and depth = 4. This caused the model accuracy to decrease slightly, from 0.83 to 0.83, so I will stick to the original model going forward.

Next I used kfold cross validation to test for overfitting. The training average accuracy was 86%, and the test average accuracy was 83%, which I wouldn't see as a major concern for overfitting.

Next, I was curious how the model ranked feature importance, so I made a list of the top 20 features

![cat top 20 features](images/cat_top_20.png)

As another way to test the performance of the model, I made an ROC curve, which shows that the model performs well.

![ROC Curve CatBoost](images/cat_AUC_Score.png)

## Discussion and Conclusion

The main question I wanted to explore with this dataset is what is the best model, which I believe is the CatBoost model because it had the highest accuracy and the best AUC curve. But all models performed well. I believe CatBoost performed the best because it better accounts for the categorical features than the other model types. 

The second question I wanted to explore was, after finding out the best model, how does it perform without any species-related features? My intention behind this was to allow people to predict which areas of the country are most likely to have chytrid fungus. The ROC curve of the model without any species-related shows that the model still performs very well. 

![ROC Curve environment CatBoost](images/cat_environment_AUC_Score.png)

It also has a different top 20 features.

![cat top 20 features environment](images/cat_environment_top_20.png)

This tells me that just the environmental variables are a good predictor of whether or not amphibians will test positive for chytrid fungus. If there are any future versions of this study, I would like to try this model again to see how well it actually holds up, because so far I only know that this model will work for this specific dataset. I also wonder how well this model will hold up with the effects of climate change, as that could impact how chytrid fungus will spread. 

There is more I would like to try with this project when I have the time. One thing I noticed when playing around with the data is that there is an issue with class imbalance. A large majority of the dataset is from one species. It's such a large majority that I didn't try and use machine learning methods to mitigate class imbalance, but this is something I'd like to try in the future. 

![Species Distribution](images/species_distribution.png)

Another thing I'd like to try with this project is to reduce the number of features in the model. I had played around with it some, by ranking the features by importance, and only keeping the most important precipitation and temperature features, and getting rid of the rest. But it reduced the accuracy of the model, and I couldn't figure out exactly which features to keep and which not to keep, so I would like to work on this more. 

