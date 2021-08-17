import pandas as pd
import os


flight_data_path="/home/inithan/Desktop/git/movie/flight_data/"


"""
##############################################################
airports* to be conside at origin and destination
filter out necessary  columns only from flight data


*for these 15 airports we have weather data
##############################################################
"""

def flight_data_pre_processing(flight_data_path):
    airports=["ATL",
    "CLT",
    "DEN",
    "DFW",
    "EWR",
    "IAH",
    "JFK",
    "LAS",
    "LAX",
    "MCO",
    "MIA",
    "ORD",
    "PHX",
    "SEA",
    "SFO",]

    #getting all json files name along with their directory for 2016 & 2017
    folders = os.listdir(flight_data_path)
    flight_files=[]
    for year in folders:
        print(year)
        files=os.listdir(flight_data_path+"/"+year+"/")
        for file in files:
            print(file)
            data=pd .read_csv(flight_data_path+"/"+year+"/"+file+"/"+file+".csv",engine='python')
            data=data[(data['Origin'].isin(airports))&(data['Dest'].isin(airports))]
            data=data[['Year','Month','DayofMonth','Origin','CRSDepTime','Dest',"CRSArrTime",'ArrDel15','ArrDelayMinutes','Flights','Distance','DistanceGroup','CRSElapsedTime']]
            flight_files.append(data)


    return(pd.concat(flight_files))

flight_data=flight_data_pre_processing(flight_data_path)
flight_data=flight_data.dropna()
flight_data.reset_index(drop=True,inplace=True)

#change 2400 time to 0
flight_data.loc[flight_data['CRSDepTime'] ==2400, 'CRSDepTime'] = 0



"""
##########################
function to create timestamp
##########################
"""
def time(rows):
    return (pd.Timestamp(rows[0], rows[1],rows[2],rows[3],rows[4]))
  
#create minutes and hour column for departure
flight_data['CRSDepTime_minutes'] = flight_data['CRSDepTime'].apply(lambda x: int(str(x)[-2:]))
flight_data['CRSDepTime_hour'] = flight_data['CRSDepTime'].apply(lambda x: int(str(x)[:-2]) if  (str(x)[:-2]).isdigit() else 0)
flight_data['CRSDepTime_time_stamp'] = flight_data[['Year','Month','DayofMonth','CRSDepTime_hour','CRSDepTime_minutes']].apply(time,axis = 'columns')


#create arrival timestamp
flight_data['CRSArrTime_time_stamp']=flight_data['CRSDepTime_time_stamp']  + pd.to_timedelta(flight_data['CRSElapsedTime'], unit='m')


# round the departure and arrival time in hourly format to match with weather format

#flight_data['CRSDepTime_hour_round'] = flight_data['CRSDepTime'].apply(lambda x: int((x- x % -100)/100) if int(str(x)[-2:])>30 else int((x- x % +100)/100))@mathamatical formula
flight_data['CRSDepTime_time_stamp_round']=flight_data['CRSDepTime_time_stamp'].apply(lambda x: x.round(freq='H'))
flight_data['CRSArrTime_time_stamp_round']=flight_data['CRSArrTime_time_stamp'].apply(lambda x: x.round(freq='H'))

flight_data=flight_data[['Origin','CRSDepTime_time_stamp_round','Dest','CRSArrTime_time_stamp_round','ArrDel15',
 'ArrDelayMinutes',
 'Flights',
 'Distance',
 'DistanceGroup',
 'CRSElapsedTime']]


#read weather data
weather_data=pd.read_csv("weather_data.csv",index_col=0)

weather_data['weather_time_stamp']=pd.to_datetime(weather_data['weather_time_stamp'])


#merge origin and destination with weather data
origin_weather_details_df=flight_data.merge(weather_data ,left_on=['Origin','CRSDepTime_time_stamp_round'] ,right_on=['airport','weather_time_stamp'] )

dest_weather_details_df=origin_weather_details_df.merge(weather_data ,left_on=['Dest','CRSArrTime_time_stamp_round'] ,right_on=['airport','weather_time_stamp'], suffixes=('_origin', '_dest') )


#save final file
dest_weather_details_df.to_csv("flight_data.csv")