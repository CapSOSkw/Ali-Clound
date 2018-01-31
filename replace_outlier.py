import pandas as pd
import numpy as np

class replace_outlier():

    def __init__(self):
        pass

    def outlier_index(self, row):
        q75, q25 = np.percentile(row, [75,25])
        iqr = q75 - q25

        lower_bound = q25 - (iqr * 1.5)
        upper_bound = q75 + (iqr * 1.5)

        outlier_index = np.where((row > upper_bound) | (row < lower_bound))[0]

        return outlier_index

    def display_outliers(self, row):
        return [row[i] for i in self.outlier_index(row)]

    def replace_outliers(self, df):
        num_rows = df.shape[0]
        for row in range(num_rows):
            row_in_df = df.ix[row, :].tolist()
            outlier_index = self.outlier_index(row_in_df)

            if len(outlier_index) != 0:
                for index in outlier_index:
                    df.ix[row, index] = np.round(np.mean(row_in_df),2)
                    print(df.ix[row, index])
            else:
                continue

        return df


if __name__ == '__main__':
    train_df = pd.read_csv("Training_new_format_mean.csv",)

    train_df_models = train_df[['model_1','model_2','model_3','model_4','model_5','model_6','model_7',
                               'model_8','model_9','model_10']]
    train_df_others = train_df[['mean', 'max', 'min','xid', 'yid', 'date_id', 'hour','target']]

    outliered_train_df_models = replace_outlier().replace_outliers(train_df_models)

    outliered_train_df = pd.concat([outliered_train_df_models, train_df_others], axis=1)

    #outliered_train_df.to_csv("Training_replace_with_mean.csv", index=False)
