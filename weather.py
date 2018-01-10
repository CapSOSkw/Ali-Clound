import pandas as pd
from collections import OrderedDict


def main():

    trainning_data = pd.read_csv("ForecastDataforTraining_201712.csv",)
    XY_data_hour = trainning_data.drop(['wind', 'model'], axis=1).iloc[0::10, :].reset_index(drop=True)

    my_dict = OrderedDict()

    for i in range(10):
        # model_df = weather_model_train.drop(['xid', 'yid', 'date_id', 'hour', 'model'], 1).iloc[i::10, :].wind.tolist()
        model_list = trainning_data['wind'].tolist()[i::10]
        my_dict['model_'+str(i+1)] = model_list

    df = pd.DataFrame(my_dict)

    temp_df = pd.concat([df, XY_data_hour], axis=1)
    # print(temp_df)

    true_data = pd.read_csv("In_situMeasurementforTraining_201712.csv", )
    true_data_rename = true_data.rename(index=str, columns={"wind": "target"})

    true_wind = true_data_rename.drop(['xid', 'yid', 'date_id','hour'], 1).reset_index(drop=True)
    # print(true_wind)

    result_df = pd.concat([temp_df, true_wind], axis=1)
    # print(result_df)

    return result_df

if __name__ == '__main__':
    # df = main()
    # df.to_csv("new_format_full_target.csv", index=False)

    df = pd.read_csv("new_format_full_target.csv", nrows=100)
    print(df)