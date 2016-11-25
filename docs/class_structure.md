## Class structure of wwdata package

###HydroData
Hydrodata is the superclass of this package, containing functions useful for all (water-related) data analyses.

Available functions include (see docstrings for more info):  
* Class functions/piped pandas functionalities  
	* set_tag()  
	* set_units()  
	* set_time_unit()  
	* head()  
	* tail()  
	* index()  
	* get_highs()  
* Formatting  
	* drop_index_duplicates()  
	* replace()  
	* set_index()  
	* to_float()  
	* to_datetime()  
	* absolute_to_relative()  
* Writing to .txt files  
	* write()  
	* write_to_WEST()  
* Data filtering  
	* delete_doubles()  
	* calc_slopes()  
	* moving_slope_filter()  
	* simple_moving_average()  
	* moving_average_filter()  
* Data (cor)relation  
	* calc_ratio()  
	* compare_ratio()  
	* get_correlation()  
	* calc_daily_profile()
* Analysis  
	* get_avg()
	* get_std()
* Plotting
	* plot_analysed()  

#### SampleBased
for data obtained in the lab by separate experiments

#### LabSensorBased
for data obtained in the lab by one single experiment; this class can contain the functions written for the project of Dries

#### OnlineSensorBased
for data obtained from field measurements

User available  functions:  
* time_to_index()
* calc_total_proporitional()
* Data filling functions  
	* add_to_filled()
	* fill_missing_interpolation()
	* fill_missing_ratio()
	* fill_missing_correlation()
	* fill_missing_standard()
	* fill_missing_modeled()

