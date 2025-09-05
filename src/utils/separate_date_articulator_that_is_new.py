'''
- When performing a code review, ensure that variable and function names use snake_case, and class names use CamelCase, following PEP 8 style guidelines.
- When reviewing functions, check if loops or conditionals can be simplified with built-in or vectorized methods (e.g., numpy, pandas, datetime, itertools) while preserving clarity and behavior.
- When reviewing a function, check that its name is appropriate and corresponds to and clearly describes its purpose.
- When reviewing a function, check that its name clearly describes its purpose and that variable names are appropriate and descriptive.
'''
from dasenima import SelectDateSpatialSlice, base_and_issue_time_declaration, generate_hourly_timestep
from utils.datasets import DataManager
from datetime import timedelta, datetime
import numpy as np

date_spatial_slice = SelectDateSpatialSlice()
def accumulated_value_update(var_name, df):
    """
    주어진 변수(var_name)에 대한 변화량을 계산하여 '_delta' 컬럼을 추가함.
    매개변수:
    - var_name (str): 변화량을 계산할 변수명.
    - df (pd.DataFrame): 변환을 수행할 데이터프레임.
    반환값:
    - pd.DataFrame: 변화량이 반영된 데이터프레임.
    """

    var_name_delta = var_name + "_delta"
    df[var_name_delta] = (df[var_name].shift(-1) - df[var_name]).fillna(0)
    df[var_name_delta] = np.where(df["leadtime"] == 9000, 0, df[var_name_delta])

    return df

def selected_time_slice(df_weatherMood):
    chosen_t0 = df_weatherMood[df_weatherMood["leadtime"]==9000]["t0"]
    (
        basetime_t0_hr_int,
        issued_time_hr_int,
        issued_date_date_int,
        issued_month_int,
        issued_time_hr_str,
    ) = base_and_issue_time_declaration(chosen_t0)

    # up until here, issued-date and hour is matched
    df_weatherMood_issueddate_all_filtered_cleanly = df_weatherMood[
        (df_weatherMood.issueddate == str(issued_date_date_int))
        & (df_weatherMood.issuedhour == issued_time_hr_str)
    ].reset_index(drop=True)

    time_slice_index_t0 = df_weatherMood_issueddate_all_filtered_cleanly[
        (df_weatherMood_issueddate_all_filtered_cleanly.basetime == chosen_t0)
    ].index[0]

    df_weatherMood_issueddate_all_filtered_cleanly = df_weatherMood_issueddate_all_filtered_cleanly.iloc[
        time_slice_index_t0 - 2 : time_slice_index_t0 + 9
    ]

    return df_weatherMood_issueddate_all_filtered_cleanly


def return_emotions_mood_weather_mixer_combinations(df_weatherMood, batch_size,num_epochs,patience ):
    df = selected_time_slice(df_weatherMood)
    sroe_code_values = df[f"{batch_size}_{num_epochs}_{patience}"]
    mood_types = df[f"{batch_size}_mood"].unique()
    weather_types = df[f"{batch_size}_weather"].unique()
    start = sroe_code_values // 100
    end   = sroe_code_values // 10

    regional_coords = date_spatial_slice(sroe_code_values)
    timestamps = generate_hourly_timestep(start, end)

    combined = [
        [mood, weather, codeNum]
        for codeNum in sroe_code_values
        for mood in mood_types
        for weather in weather_types
    ]

    all_timesteps = generate_monthly_timestamps(start, end)

    return combined, all_timesteps

def generate_monthly_timestamps(start_timestamp, end_timestamp):
    # datetime 으로 변환
    start_timestamp = str(start_timestamp)
    end_timestamp = str(end_timestamp)
    start_date = datetime.strptime(start_timestamp, "%Y%m%d%H")
    end_date = datetime.strptime(end_timestamp, "%Y%m%d%H")

    # 각 개월 수 마다로
    monthly_timestamps = {}

    # 각 매시간 (everyhour) 루핑
    current_time = start_date
    while current_time <= end_date:
        month_key = current_time.strftime("%Y%m")  #  YYYYMM
        timestamp_int = int(current_time.strftime("%Y%m%d%H%M%S"))  # YYYMMDDHHMMSS

        # 해당 달이 새로 시작되면, dict에 새로운 키를 만든다
        if month_key not in monthly_timestamps:
            monthly_timestamps[month_key] = []

        # 해당 알맞는 개월 key에 element로 Dict에 포함
        monthly_timestamps[month_key].append(timestamp_int)

        # 아음 시간으로
        current_time += timedelta(hours=1)

    return monthly_timestamps