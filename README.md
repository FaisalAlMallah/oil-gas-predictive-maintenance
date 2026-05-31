# oil-gas-predictive-maintenance
Try the project: https://oil-gas-predictive-maintenance.vercel.app/	

Project Title: Oil & Gas Predictive Maintenance: Failure Prediction Within 24 Hours 

Project Summary: The goal is to predict whether equipment such as pumps or machines may fail within the next 24 hours using sensor and maintenance data. 

Defining the problem: 

Predicting when oil and gas products need maintenance (pumps etc)

Success would look like if we can help improve safety measures, unplanned downtime, reduce costs 

Binary classification ML problem , target class is machines may fail within the next 24 hours

Metrics that matter: accuracy, confusion matrix, f1 score

Data description:

https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance	
Records: 24,042
Features: 15 columns
Machines: 20 unique units
Machine Types: CNC, Pump, Compressor, Robotic Arm
Missing Values: 4,022 (~1.12%)
Size: 2.17 MB
Data Type: Synthetic, timestamped operational data


EDA/ preparing Data: 
Adjusted Data types to what's needed
Handled Null values
Removed leakage columns 
Distribution of Target column ( failure_within_24hr 0 1 ) SD = 0.36
Feature scaling 
One hot encoding 


Base Line model:
The Logistic Regression baseline achieved 94.85% test accuracy and an F1-score of 0.81 for predicting failures within 24 hours. The model performs similarly on training and test data, suggesting no major overfitting. However, the model still misses 130 actual failures, so improving recall/F1-score for the failure class should be a focus in future models.


Model:
The XGBoost model achieved 98.77% test accuracy and an F1-score of 0.96 for predicting failures within 24 hours. It performed much better than the Logistic Regression baseline, reducing missed failures from 130 to only 20. However, the training score is almost perfect, so slight overfitting may be present and should be checked with cross-validation or tuning.
 




## Run locally

Start the local web server:

```bash
python local_server.py
```

Then open:

```bash
http://127.0.0.1:8000
```

This serves both the frontend and the `/api/predict` endpoint on the same origin.


