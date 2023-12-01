# Fire suppression study

Analysis of fire suppression study data from the US Forest Service

## Data

### Raw data

The dataset has been extracted from the SIT209 dataset, on a time period of 2015-2018.
The raw data should be in the `data/raw` folder (4 CSV files).

### Cleaned data

Each of these datasets has been individually cleaned: date cleaning, merge some categorical encodings (cause_identifier) throughout years.The code for cleaning is in their respective respective notebooks. Cleaned columns have lowercase names.
Cleaned data should be in the `data/cleaned` folder (3 files).

### Preprocessed

Finally, these three datasets have been merged and preprocessed for further analysis. At this step, no imputation has been performed.
Preprocessed data is available in the `data/preprocessed` folder (1 file).
