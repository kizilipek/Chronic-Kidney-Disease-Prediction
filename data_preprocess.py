from random import SystemRandom
import pandas as pd 
import numpy as np 
import os
from sklearn.preprocessing import MinMaxScaler
import argparse 
import gc 
import pickle
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--add-meds',default=False, help='add meds data or not: True/False')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--raw-data-dir', default='./raw_data')
parser.add_argument('--processed-data-dir',default='./processed_data')

args = parser.parse_args()
scaler = MinMaxScaler()


def interpolate_df(df):
    interpolated_df = pd.DataFrame()
    for i in range(0,300):
        tmp = pd.DataFrame({'id':[i]*len(list(range(0,700))),'time':list(range(0,700))}) 
        new_df = pd.merge(tmp,df,how='left',on=['id','time'])
        new_df['value'] = new_df['value'].interpolate(method='linear', limit_direction='forward', axis=0)
        interpolated_df = interpolated_df.append(new_df)
        del tmp
        del new_df
        gc.collect()
    return interpolated_df


if __name__ == '__main__':

    data_experiment_id = int(SystemRandom().random()*100000)
    print(args, data_experiment_id)
    seed = args.seed
    np.random.seed(seed)

    
    longtidunal_data_files = ['T_creatinine.csv',
    'T_DBP.csv',
    'T_glucose.csv',
    'T_HGB.csv',
    'T_ldl.csv',
    'T_SBP.csv']
    
    dataframe_collection={}
 
    # read the biomarkers data 
    for file in longtidunal_data_files:
        if file.endswith('csv'):
            filename = file.split('.')[0].lower()
            dataframe_collection[filename] = pd.read_csv(os.path.join(args.raw_data_dir,file))

    # check what is the time interval each biomarker is being measured 
    for dfname, df in dataframe_collection.items():
        print(f"{dfname} minimum time: {df['time'].min()} and maximum time:{df['time'].max()}")

    interpolated_data_file = 'interpolated_data.json'

    if os.path.exists(os.path.join(args.processed_data_dir, interpolated_data_file)):
         with open(os.path.join(args.processed_data_dir, interpolated_data_file), 'rb') as file:
            dataframe_collection  = pickle.load(file)
    else:
        for dfname, df in dataframe_collection.items():
            # normalize the data by using min max scaling 
            xmin = df['value'].min()
            xmax = df['value'].max()
            df['value'] = df['value'].apply(lambda x: (x - xmin) / (xmax - xmin))
            # interpolate each data frame for missing data 
            interpolated_df = interpolate_df(df)
            interpolated_df = interpolated_df.rename(columns={'value': dfname.split('_')[1]}) 
            dataframe_collection[dfname] = interpolated_df 

        os.makedirs(args.processed_data_dir)
        with open(os.path.join(args.processed_data_dir, interpolated_data_file), 'wb') as file:
            pickle.dump(dataframe_collection, file)

    # merge all the data into single dataframe 
    frames=[v for k,v in dataframe_collection.items()]
    # hgb data is measured too many times compare to the other data, thus we truncate the hgb data 
    dataframe_collection['t_hgb'] =  dataframe_collection['t_hgb'][dataframe_collection['t_hgb'].time <700]
    data = frames[0]
    for frame in frames[1:]:
        data = data.merge(frame, how='left', on=['id','time'])
    data.to_csv(os.path.join(args.processed_data_dir , 'data.csv'), index=False)

    
    # combine meds data 
    if args.add_meds:
        print('preparing_medical_data')
        meds = pd.read_csv(os.path.join(args.raw_data_dir, 'T_meds.csv'))

        # get rid of the time values which are negative
        meds.loc[meds.start_day<0, 'start_day'] = 0 
        meds.loc[meds.end_day<0, 'end_day'] = 0

        new_meds = []

        # convert medication summary into daily time series data based on the start and end day 
        for id,drug in zip(meds[['id','drug']].drop_duplicates()['id'],meds[['id','drug']].drop_duplicates()['drug']):
            for ind, row in meds[(meds['id']==id)&(meds['drug']==drug)].iterrows():
                tmp = pd.concat([row[:-2].to_frame().T]* (row['end_day'] - row['start_day'] +1), ignore_index=True)
                tmp['time'] = list(range(row['start_day'], row['end_day']+1))
                new_meds.append(tmp)

        # convert drugs information into column features          
        meds_df = pd.concat(new_meds, ignore_index=True)
        meds_df = meds_df.pivot_table(index=['id','time'], columns ='drug', values = 'daily_dosage').reset_index()
        meds_df = meds_df.fillna(0)
        # normalize the medication data
        meds_df[meds_df.columns[2:]] = scaler.fit_transform(meds_df[meds_df.columns[2:]])

        # merge longitudinal and medication data 
        df_merged = data.merge(meds_df, how='left', on=['id','time'])
        df_merged = df_merged.fillna(0)
        df_merged.to_csv(os.path.join(args.processed_data_dir , 'data.csv'), index=False)


