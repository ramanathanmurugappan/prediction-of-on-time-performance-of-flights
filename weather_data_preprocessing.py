import pandas as pd
import os


weather_data_path="/home/inithan/Desktop/git/movie/weather/"

#getting all json files name along with their directory for 2016 & 2017
folders = os.listdir(weather_data_path)
weather_files=[]
for folder in folders:
    #print(folder)
    weather_files=weather_files+([weather_data_path+folder+'/'+i for i in os.listdir(weather_data_path+folder) if any(j in i for j in ['2016','2017'])])


""""
###################################################################################################
weather_data_pre_processing(*all json weather files with directory in list)

* remove url and unwanted dict inside dictionary
* add airport,year,month,day

return weather data in dataframe
###################################################################################################
"""
def weather_data_pre_processing(weather_files):
    columns_required=['windspeedKmph',
    'FeelsLikeF',
    'winddir16Point',
    'FeelsLikeC',
    'DewPointC',
    'windspeedMiles',
    'DewPointF',
    'HeatIndexF',
    'cloudcover',
    'HeatIndexC',
    'precipMM',
    'WindGustMiles',
    'pressure',
    'WindGustKmph',
    'visibility',
    'weatherCode',
    'tempC',
    'tempF',
    'WindChillF',
    'WindChillC',
    'winddirDegree',
    'humidity',
    'time']#selected columns from json
    master_data=[]
    for file in weather_files:
        data=pd.read_json(file)
        for day in range(len(data['data']['weather'])):
        #print(day)
            for hour in range(len(data['data']['weather'][day]['hourly'])):
                temp_dict={ key:value for (key,value) in data['data']['weather'][day]['hourly'][hour].items() if key in columns_required}
                temp=file[40:]
                #print(temp[:temp.find("/")])
                temp_dict['airport']=temp[:temp.find("/")]
                temp_dict['Year']=int(temp[temp.find("/")+1:temp.find("/")+5])
                temp_dict['Month']=int(temp[temp.find("/")+6:-5])
                temp_dict['DayofMonth']=day+1
                master_data.append(pd.DataFrame([temp_dict]))

    
    return(pd.concat(master_data))

weather_data=weather_data_pre_processing(weather_files)

"""
##########################
function to create timestamp
##########################
"""

def time(rows):
    return (pd.Timestamp(rows[0], rows[1],rows[2],rows[3]))

weather_data['hour'] = weather_data['time'].apply(lambda x: int(str(x)[:-2]) if  (str(x)[:-2]).isdigit() else 0)

weather_data['weather_time_stamp'] = weather_data[['Year','Month','DayofMonth','hour']].apply(time,axis = 'columns')



weather_data[['windspeedKmph',
 'FeelsLikeF',
 'winddir16Point',
 'FeelsLikeC',
 'DewPointC',
 'windspeedMiles',
 'DewPointF',
 'HeatIndexF',
 'cloudcover',
 'HeatIndexC',
 'precipMM',
 'WindGustMiles',
 'pressure',
 'WindGustKmph',
 'visibility',
 'weatherCode',
 'tempC',
 'tempF',
 'WindChillF',
 'WindChillC',
 'winddirDegree',
 'humidity',
 'airport',
 'weather_time_stamp']].to_csv("weather_data.csv")
