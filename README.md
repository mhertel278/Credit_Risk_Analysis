# Credit_Risk_Analysis

## Resources
python 3.7.9; pandas, sklearn and imblearn libraries; jupyter notebook; csv of loan statistics for 2019 first quartet 

## Overview

The purpose of this project is use data on loans from 2019 to develop a machine learning model to predict if a future loan applicant is high risk or low risk to default on the loan.  

Using pandas I extracted and transforme the data from the csv, converting the target 'loan_status' column values into either 'high risk' or 'low risk' depending on whether payments toward the loan were behind in some way, or if they were current.  I then converted featurescolumns with text--such as current home ownership status and any hardship flags--into numerical values that could be interpreted by the machine learning algorithm.  I then separated the data into columns for features, and a target column containing a status of 'high risk' or 'low risk'.  I then split the data into training and testing sets.

Because the classes of low risk and high risk were so imbalanced, I tested several over-sampling and under-sampling models to get their accuracy, precision, and recall scores.

I also then tested two ensemble classifier models to determine their accuracy, precision, and recall scores.

## Results

### Resampling Models Results

- Random Oversampling Model Results

    - Accuracy Score: 0.66
    
    ![ros accuracy](Resources/1_ros_accuracy.png)

    - High Risk Precision Score: 0.01

    - Low Risk Precision Score: 1.00

    - High Risk Recall Score: 0.68

    - Low Risk Recall Score: 0.64

    ![ros_report](Resources/1_ros_report.png)

- Oversampling with SMOTE Model Results

    - Accuracy Score: 0.655
    
    ![smote accuracy](Resources/2_smote_accuracy.png)

    - High Risk Precision Score: 0.01

    - Low Risk Precision Score: 1.00

    - High Risk Recall Score: 0.61

    - Low Risk Recall Score: 0.70

    ![smote_report](Resources/2_smote_report.png)

- Undersampling with Clustered Centroid Model Results

    - Accuracy Score: 0.545
    
    ![cc accuracy](Resources/3_cc_accuracy.png)

    - High Risk Precision Score: 0.01

    - Low Risk Precision Score: 1.00

    - High Risk Recall Score: 0.69

    - Low Risk Recall Score: 0.40

    ![cc_report](Resources/3_cc_report.png)

- Combined Over/Undersampling with SMOTEENN Model Results

    - Accuracy Score: 0.642
    
    ![smoteen accuracy](Resources/4_smoteen_accuracy.png)

    - High Risk Precision Score: 0.01

    - Low Risk Precision Score: 1.00

    - High Risk Recall Score: 0.71

    - Low Risk Recall Score: 0.57

    ![smoteenn_report](Resources/4_smoteen_report.png)

### Ensemble Classifier Models Results
    
- Balanced Random Forest Classifier Model Results

    - Accuracy Score: 0.789
    
    ![rfor accuracy](Resources/5_rfor_accuracy.png)

    - High Risk Precision Score: 0.03

    - Low Risk Precision Score: 1.00

    - High Risk Recall Score: 0.70

    - Low Risk Recall Score: 0.87

    ![rfor_report](Resources/5_rfor_report.png)

- Easy Ensemble Classifier Model Results

    - Accuracy Score: 0.931
    
    ![ada accuracy](Resources/6_adaboost_accuracy.png)

    - High Risk Precision Score: 0.09

    - Low Risk Precision Score: 1.00

    - High Risk Recall Score: 0.92

    - Low Risk Recall Score: 0.94

    ![ada_report](Resources/6_adaboost_report.png)