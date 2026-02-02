# PV Forecasting (1-12 Hour Horizons)

This PV forecasting project uses machine learning to produce ac-power prediction models for 4 different sites operated by NREL in Boulder, Colorado.

## Overview
Since solar output is at the mercy of environmental factors and highly variable, it is very useful for utilities and consumers to be able predict solar output for a given system. This project uses data from the PVDAQ data bank, which contains real data from real sites across America. In addition, it pulls historical forecast data from HRRR (High-Resolution Rapid Refresh), a short-term weather model from NOAA, to provide historical weather context for recorded PVDAQ values. Through this approach, a machine learning model can learn how to utilize weather forecasts, along with other site data, to predict future ac power output. 

## Environment Setup

Run the following commands to set up the environment.

* *conda env create -f environment.yml*
* *conda activate forecasting-env*

## Data Extraction
1. PV Data is pulled programatically from publically available PVDAQ data using the pvdaq-access PyPi library. This project pulled data from systems 4,10,50,51. The resulting data is from all years and contains unecessary metrics which will be cleaned later.

* *Run: python extract_pvdaq_data.py*

2. Weather data is pulled programmatically from publically available HRRR historical weather data using Herbie (herbie-data) on PyPi. The script pulls forecasted values for t2m (temperature two meters from ground), tcc (total cloud cover), dswrf (downward short-wave radiation flux) from forecast horizons 1-12 for every hourly timestamp between January 2019 - February 2023. This time period was chosen due to a combination of consistent PVDAQ data and HRRR coverage. The extraction process is very consuming which is why this project ran this section on a powerful EC2 instance to allow for parallelization.

* *Run: python extract_hrrr_data.py*

#### Data Description
- Target variable:
  - `ac_power_norm` (normalized AC power output)
- Key features:
  - Time features: hour_sin, hour_cos, doy_sin, doy_cos (day   and hour encoded cyclically)
  - Solar geometry features: cosine of solar zenith
  - System identifiers
  - Weather features: t2m (temperature two meters from ground), tcc (total cloud cover), dswrf (downward short-wave radiation flux)
  -Persistence features: ac_power_on_issue (what the ac power at the time you are making the prediction)

## Data Processing

3. Individual system data is split across thousands of csv files when originally extracted. They need to be merged into one. In the process, only relevant dates and columns are kept, with the most important being ac power. Unfortunately, the systems often have different column names for ac power, often with multiple columns for ac power present in one system. Therefore, manual review of the systems is needed to determine the correct ac power column to pull, with this column name being passed as a script's parameter. Furthermore, system_capacity needs to be passed in as a positional argument as well to create a normalized ac power column for all systems.

* *Use: python process_individual_systems.py <ac_power_column_name> <system_id> <system_capacity [Watts]>"*
* *Ex:  python process_individual_systems.py ac_power__315 4 1000 # (this will process data for system 4, which records power in ac_power__315 column and has a system capacity of 1kW)*

**The following commands were used for the 4 systems in this project:**
* python process_individual_systems.py ac_power__315 4 1000
* python process_individual_systems.py ac_power__423 10 1120
* python process_individual_systems.py ac_power__752 50 6000
* python process_individual_systems.py ac_power__773 51 6000

4. After all systems are processed, the systems need to be combined into one csv.

* *Run: combine_pvdaq_systems.py*

5. Now, the pvdaq data will be merged with HRRR data. The following script merges data and calculates model features, like sun geometry, hour and day cyclical encodings, and one-hot system encodings. It also drops incomplete rows, nightime rows, and clips/filters certain columns. Lastly, it splits all horizons into their own folder because each horizon will get its own model.

* *Run: clean_and_merge.py*

6. The data for each horizon will be split into training/validation/testing splits based on time period. 

* *Run: create_splits.py*

## Training

XGBoost Model
    Strategy:
    The XGBoost model uses gradient boosted decision trees to predict normalized AC power output for each forecast horizon independently. This is a direct forecasting approach where each horizon (1-12 hours ahead) is treated as a separate prediction task.
 
    Training: Uses early stopping with validation monitoring (50 rounds patience)
    Hyperparameters:

    Learning rate: 0.05
    Max depth: 6
    Subsample: 0.8
    Column subsample: 0.8
    Up to 1000 boosting rounds

LSTM (Long Short-Term Memory) Model
    Strategy:
    The LSTM model uses recurrent neural networks to capture temporal dependencies in solar power generation. It processes sequences of historical data (lookback window) to predict the next time step, explicitly modeling the time-series nature of the problem.

    Architecture:

    System embedding layer (4 dimensions) to capture system-specific characteristics
    Single LSTM layer (64 hidden units)
    Fully connected output layer
    Combines LSTM hidden state with system embedding for prediction

    Sequence Processing:

    Lookback window: 6 hours of historical data
    Enforces strict 1-hour continuity (no gaps in time series)
    Separate sequences per solar system

    Training:

    Batch size: 256
    Epochs: 30
    Optimizer: Adam (learning rate: 0.001)
    Loss: Mean Squared Error (MSE)`

7. Models can be trained with any of the scripts in the train folder. Resulting trained models will be placed in the models directory.

* *Run: (ex. python xgboost_main.py)*

## Testing

8. Trained models can be tested with either the test_lstm.py or test_xgboost.py scripts. All xgboost scripts (baseline scripts + xgboost_main) must be tested with test_xgboost.py. All lstm scripts (lstm_main.py) must be tested with test_lstm.py. Both testing scripts take in the directory of the trained model as a positional argument, and will return a csv in the results/[input_dir] directory with model performance information.

* *Run: python test_lstm.py <trained_model_dir>*
* *Example:  python test_lstm.py lstm_main*

## Visualize

9. It is useful to compare the performance of different models visually. The following script produces a png and pdf file showing the RMSE across horizons for the different models. The results of this project indicate that the LSTM model produces the most accurate predictions, with RMSE stable at around 0.14-0.16 for all horizons (see results/plots).

* *Run: python visualize.py*

