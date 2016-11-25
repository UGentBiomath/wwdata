#Proposed class structure of hydropy v2 package

##HydroData
This superclass can contain some current, generally applicable hydropy functions and some functions written for the project of Dries
### SampleBased
for data obtained in the lab by separate experiments
### LabSensorBased
for data obtained in the lab by one single experiment; this class can contain the functions written for the project of Dries
### OnlineSensorBased
for data obtained from field measurements; this class can contain a lot of the current hydropy functions that are more aimed at hydrological timeseries.
