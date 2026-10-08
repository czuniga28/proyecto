# Data Leakage and Dataset Limitations Report

## Limitations

### Lack of Patient ID
The dataset does not contain a unique Patient ID (`patientId`). This poses a limitation when grouping the data to prevent leakage. The same person could potentially have multiple different exams in the dataset (for example, imaging both knees or having a follow-up scan). 

Because `examId` is the best grouping available, we use it to group the volumes and stratify the splits. While grouping by `examId` ensures that both series of a single exam (which belong to the same knee) are kept together in the same split, the lack of `patientId` remains a potential data leakage risk. A model could theoretically be evaluated on a knee (or patient) it has already seen if that patient had multiple distinct exams. 

This is a known limitation of the current splitting strategy and must be acknowledged when evaluating the model's generalizability.
