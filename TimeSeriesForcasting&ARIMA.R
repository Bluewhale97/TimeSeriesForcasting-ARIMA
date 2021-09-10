#1. functions for time-series analysis
ts()#in stats package, creats a time-series object
plot()#in gprahics package, polots a time series
start()#in stats, returns the starting time of a time series
end()#in stats, returns the ending time of a time series
frequency()#in stats, returns the period of a time series
window()#in stats, subsets a time-series object
ma()#in forecast package, fits a simple moving-average model
stl()#in stats, decomposes a time series into seasonal, trend and irregular components using loess
monthplot()#in stats, plots the seasonal components of a time series
seasonplot()#in forecast, generates a season plot
HoltWinters()#in stats, fits an exponential smoothing model
forecast()#in forecast, forecasts future values of a time series
accuracy()#in forecast, reports fit measures for a time-series model
ets()#in forecast,fits an exponential smoothing model. include the ability to automate the selection of the model
lag()#in stats, returns a lagged version of a time seires
Acf()#in forecast, estimates the autocorrelation function
Pacf()#in forecast, estimates the partia autocorrelation function
diff()#in base package, returns agged and iterated differences
ndiffs()#in forecast, determines the level of differencing needed to remove trends in a time series
adf.test()#in tseries package, computes an augmented dickey-fuller test that a time series is stationary
arima()#in stats, fits autoregressive integrated moving-average models
Box.test()#in stats, computes a Ljung-Box test that the residuals of a time series are independent
bds.test()#in tseries package, computes the BDS test that a series consists of independent, identically distributed random variables
auto.arima()#in forecast, automates the selection of an ARIMA model

#2. creating a time-series object that contains the informtaion about its periodicity

#using ts() function
myseries<- ts(data,start=,end=,frequency=) #where myseries is the time-series object, data is anumeric vector containing the observations
#start specifies the series start time, end specifies the end time(optional) and frequency indicates the number of observayions per unit time
#frequncy=1 for annual data, frequency=12 for monthly data, and frequency=4 for quarterly data


sales <-c(18,33,41,7,34,35,24,25,24,21,25,20,
          22,31,40,29,25,21,22,54,31,25,26,35)
tsales <-ts(sales, start=c(2003,1), frequency=12)
tsales
plot(tsales)
start(tsales)

end(tsales)

frequency(tsales)

tsales.subset <-window(tsales, start=c(2003,5), end=c(2004,6))
tsales.subset

#we can modify plot using plotting parameters within 


#3. smoothing and seasonal decomposition

#a. smoothing with simple moving averages

#time series typically have a significant irregular or error component
#in order to discern any patterns in the data
#we will freqently want to plot a smoothed curve that damps down these fluctuations
#one of the simplest methods of smoothing a time series is to use simple moving averages

#for example, each data point can be replaced with the mean of that observation and one observation before and after it, this is called a centered moving average
#a centered moving average is defined as St=(Yt-q+...+Yt+... Yt+q)/(2q+1)
#where St is the smoothed value at time t and k=2q+1 is the number of observations that are averaged. The k value is usually chosen to be an odd number
#By necessity, when using a centered moving average, we lose the (k-1)/2 observations at each end of the series


#SMA() in the TTR package and rollmean() in the zoo package and ma() in the forecast package can provide a simple moving average


#use ma() function to smooth the Nile time series that comes with base R installation
install.packages("forecast")
library(forecast)
opar<-par(no.readonly=T)
par(mfrow=c(2,2))
ylim<-c(min(Nile),max(Nile))
plot(Nile, main="Raw time series")
plot(ma(Nile,3), main="Simple MOving Average (k=3)", ylim=ylim)
plot(ma(Nile,7), main="Simple MOving Average (k=7)", ylim=ylim)
plot(ma(Nile,15), main="Simple MOving Average (k=15)", ylim=ylim)
par(opar)
#as k increases, the plot becomes increasingly smoothed
#the challenge is to find the value of k that highlights the major patterns in the data, without under- or over-smoothing
#this is more art than science, we will probably want to try several values of k before settling on one

#b. seasonal decomposition: can be used to examine both seasonal and general trends
#time series data that have a seasonal aspect can be decomposed into a trend component, a seasonal component and an irregular component

#trend component captures changes in level over time
#seasonal component captires cyclical effects due to the time of year
#the irregular(or error) component captures those influences not described by the trend and seasonal effects
Yt=Trendt+Seasonalt+Irregulart #this ia an additive model, the components sum to give the values of the time series

Yt=Trendt*Seasonalt*Irregulart #this is a multiplicative model

#notice how the variability is proportional to the leve: as the level increases, so does the variability
#this amplification based on the current level of the series strongly suggests a multiplicative model

#in many instances, the multiplicative model is more realistic

#popular method for decomposing a time series into trend, seasonal and irregular components is by loess smoothing
stl(ts, s.window=, t.window=) #where ts is the time series to be decomposed, s.window controls how fast the seasonal effects can change over time, t.window controls how fast the trend can change over time
#smaller values allow more rapid change
#setting s.window="periodic" forces seasonal effects to be identical across year
#only the ts and s.window parameters are required
#stl() function only apply to handle additive models
#but multiplicative models can be transformed into additive models using a log transformation

plot(AirPassengers)#the time series is plotted and transformed
lAirPassengers <-log(AirPassengers)
plot(lAirPassengers, ylab="log(AirPassengers)")

fit <-stl(lAirPassengers, s.window="period")
plot(fit)

#the graph shows the seasonal components have been constrained to remain the same across each year(using the s.window="period" option)
#the trend is monotonically increasing
#the grey bars on the right are magnitude guides, each bar represents the same magnitude

#the object returned by the stl() function contains a component called time.series that contains the trend, season and irregular portion of each observation
#in this case, fit$time.series is based on the logged time series
#exp(fit$time.series) converts the decomposition back to the original metric

#monthplot() 
#seasonplot() 

par(mfrow=c(2,1))
library(forecast)
monthplot(AirPassengers, xlab="", ylab="")#along with average of each subseries
seasonplot(AirPassengers, year.labels="TRUE", main="")
fit$time.series

#4. exponential forecasting models
#single exponential model(simple exponential model) fits a time serie that has a constant level and an irregular component at time i but has neither a trend nor a seasonal component

#double exponential moedl(also called Holt exponential smoothing) fits a time series with both a level and a trend

#a triple exponential model(Hot-Winters exponential smoothing) fits a time series with level, trend and seasonal components

HoltWinters()#in the base installation 
ets(ts, model="ZZZ")#comew with the forecast package, has more options and is generally more powerful
#where ts is a time series and the model is specified by three letters
#the first letter denotes the erroe type
#the second denotes the trend type and the third for the seasonal type
#Allowable letrters are A for additive, M for multiplicative, N for none and Z for automatically selected

#Type   Parameters fit          Functions
#simple      level            ets(ts, model="ANN")  or ses(ts)
#double       level, slope       ets(ts, model="AAN") or holt(ts)
#triple      level, slope, seasonal  ets(ts, model="AAA") or hw(ts)


#5. simple exponential smoothing
#using a weighted average of existing time series values to make a short term prediction of future values
#the weights are chosen so that observations have an exponentially decreasing impact on the average as we can go back in time

Yt=level + irregulart
#the prediction at time Yt+1 called the 1-step ahead forecast is written as
Yt+1=c0Yt+c1Yt-1+c2Yt-2+c2Yt-2+... #where ci=alpha*(1-alpha)^i, alpha between (0,1)
#alpha weights for the rate of decay 
#the closer alpha is to 0, the more weight is given to past obsevations

#the actual value of alpha is usually chosen by computer in order to optimize a fit cirterion
#a common fit criterion is the sum of squared errors between the actual and predicted values


library(forecast)
fit<-ets(nhtemp, model="ANN")
fit
#there is no obvious trend and the yearly data lack a seasonal component, so the simple exponential model is a reasonal palce to start

forecast(fit,1)

plot(forecast(fit,1), xlab="Year",
     ylab=expression(paste("Temperature (", degree*F,")",)),
     main="New Haven Annual Mean Temperature")
accuracy(fit)

#ANN mode fits the simple exponential model to the nhtemp time series
#alpha=.18 is low and indicates that distant as well as recent observations are being considered in the forecast
#this value is automatically chosen to maximize the fit of the model to the given data set

#6. accuracy() of forecast package

#is used to predict the time series k steps into the future
#format is forecast(fit,k)

#the 1-step ahead forecast for this series is 51.9 degree with a 95% CV(49.7 to 54.1)
#the 80% and 95% CV are plotted both

#the forecast package also provides an accuracy() function that displays the mos tpopular predictive accyract measures for time series forecasts
#predictive accuracy measures
#

#Measure                       Abbreviation        Definition
#Mean error                        ME                 mean(et)
#Root mean squared error         RMSE               sqrt(mean(et^2))
#Mean absolute error               MAE              mean(|et|)
#Mean percentage error           MAPE                mean(|100*et/Yt|)
#Mean absolute scaled error      MASE                mean(|qt|) 

#et represents the error or irregular component of each observation (Yt-Yi)
#qt=et/(1/(T-1)*sum(|yt-yt-1)), T is the number of observations, and the sum goes from t=2 to t=T


#the mean error and mean percentage error may not be that useful, because positive and negative erros can cancel out
#RMSE gives the square root of the mean square error
#the MAPE reports the error as a percentage of the time series values, it is unit-less and can be used to compare prediction accuracy across time series
#but it assumes a measurement scale with a true zero point with a true zero point(but the Fahrenheit scale has no true zero so we cant use this)

#the MASE is the most recent accuracy measure and is used to compared the forecast accuracy across time series  on different scales


#7. Holt and Holt-Winters exponential smoothing

#Holt exponential smmothing can fit a time series that has an overall level and a trend (slope) at time t
Yt=level+slope*t+irregulart
#alpha controls the exponential decay for the level
#beta smoothing parameter controls the exponential decay for the slope
#beta also ranger from 0 to 1, with larger values giving more weight to recent observations

#Holt-Winters exponential smoothing approach can be used to fit a time series that has an overall level, trend and a seasonal component at time t
Yt=level+slope*t + st + irregulart #where st represents the seasonal influence at time t

#a gamma smoothing parameter controls the exponential decay of the seasonal component range from 0 to 1, larger values give more weight to recent observations in calculating the seasonal effect

library(forecast)
fit <-ets(log(AirPassengers), model="AAA")
fit
accuracy(fit)

pred<- forecast(fit,5)
pred

plot(pred, main="Forecast for Air Travel")

pred$mean <-exp(pred$mean)#point forecasts
pred$lower <-exp(pred$lower)#80% and 95% lower confidence limits
pred$upper <-exp(pred$upper)#80% and 95% upper confidence limits
p<-cbind(pred$mean, pred$lower, pred$upper)
dimnames(p)[[2]]<-c("mean","Lo 80","Lo 95","Hi 80","Hi 95")
p

#the smoothing parameters for the level .82 , trend .0004 and seasonal components .012 are given
#low value for the trend .004 doesnt mean there is no slope, it indicates that the slope eastimated from early observations didnt need to be updated

#exp() is used to return the predictions to the original scale, and cbind for creating a single table


#8. ets() function and automate forecasting
#fit exponentia models that have multiplicative components, add a dampening component, and perform automated forecasts

ets(AirPassengers, model="MAM")
hw(AirPassengers, seasonal="multiplicative") #these two are equivalent
#in this case the trend remains additive but the seasonal and irregular components are to be multiplicative
#by using a multiplicative model in this case, the accuracy statistics and forecasted values are reported in the original metric

#the ets() can also fit a damping component, time series predictions often assume that a trend will continue up forever
#a damping component forces the trend to a horizontal asymptote over a period of time
#in many cases, a damped model makes morerealistic predictions 

#we can invoke the ets() to automatically select a best-fitting model for the data

library(forecast)
fit<-ets(JohnsonJohnson)
fit

plot(forecast(fit), main="Johnson & Johnson Forecast",
     ylab="Quarterly Earnings ")

#because no model is specified, the software performs a search over a wide array of models to find one that minimizes the fit criterion(log-likelihood by default)
#selected model is one tha tahs multipicative trend, seasonal and error components

#flty parameter sets the line type for the forecast line(dashed in this case)


#8. ARIMA forecasting models
# in the autoregressive integrated moving average ARIMA approach to forecasting, predicted values are a linear function of recent actual values and recent errors of prediction(residuals)
#ARIMA models is complex, we first discuss ARIMA models for non-seasonal time series

#9. prerequisite concepts for ARIMA
#lag, when we lag a time series, we shift it back by a given number of observations, lag 1 is the time series shifted one position to the left
#lag(ts,k) where ts is the time series and k is the number of lags

#autocorrelation, measures the way observations in a time series relate to each other. ACk is the correlation between a set of observation Yt and observations k periods earlier Yt-k
#so AC1 us rge cirrekation between the Lag1 and Lag0 time series
#ploatting these correlations (AC1, AC2,...,ACk)produces an autocorrelation function ACF plot, which is used to select appropriate parameters for the ARIMA model and to assess the fit of the final model

#an ACF plot can be produced with the acf()function in the stats package or the Acf() in the forecast package, Acf() is easier to read

#Partial autocorrelatioon is the correlation between Yt and Yt-k with the effects of all Y values between the two(Yt-1, Yt-2,..., Yt-k+1) removed
#partial autocorrelations can be plotted for multiple values of k. 
#the PACF plot can be generated with either the pacf() in the stats or the Pacf() in the forecast package
#Pacf() is preferrd due to its formatting, this function called Pacf(ts), where ts is the time series to be assessed

#PCAF plot also is used to determine the most appropriate parameters for the ARIMA model

#ARIMA models are designed to fit stationary time series(or time series that can be made stationary)
#the statistical properties of the series dont change over time
#for example, the mean and variance of Yt are constant, additionally, the autocorrelations for any lag k dont change with time

#it may be necessary to transform the values of a time series in order to achieve constant variance before proceeding to fitting an ARIMA model
#log transformation is often useful here

#differencing, many non-stationary time series can be made stationary through differencing. 
#In differencing, each value of a time series Yt is replaced with Yt-1-Yt
#Differencing, each value of a time series Yt is replaced with Yt-1-Yt
#differencing a time series once removes a linear trend
#difference it a second time removes a quadratic trend
#a thrid time removes a cubic trend
#it is rarely necessary to difference more than twice
#diff(ts, differences=d) where d indicates the number of times the time series ts is differenced
#ndiffs(ts) function in the forecast package can be used to help determine the best value of d.


#to sum up
#ACF and PCF plots are used to determine the parameters of ARIMA models
#stationarity is an important assumption and transformations and differenceing are used to help achive stationarity
#with these, we can now trun to fitting models with an autoregressive(AR)component, a moving averages(MA) component or both components(ARMA)
#we also can examine ARIMA  models that include ARMA components and differencing to achieve stationarity(integration)

#10. ARMA and ARIMA models

#in an autoregressive model of order p, each value in a time serie s is predicted from a linear combination of the previous p values
AR(p):Yt=u+Beta1*Yt-1+Beta2*Yt-2+...+Betap*Yt-p+Sigmat
#where Yt is a given value of the series, u is the mean of the series, the Betas are the weights and Sigmat is the irregular component

#in a moving average model of order q, each value in the time series is predicted from a linear combination of q previous errors

#combining the two approaches yield an ARMA MODEL 

#an ARIMA(p,d,q)model is amoedl in which the time series has been differenced d times, and the resulting values are predicted from the previous p actual vaulues and q previous errors
#the predictions are un-differenced or integrated to achieve the final prediction

#11. steps in ARIMA modeling
#a.ensure that the time series is stationary
#b. identify a reasonable model or models(possible values of p and q)
#c. fit the model
#d. evaluate the model's fit, including statistical assumptions and predictive accuracy
#e. make forecasts

library(forecast)
library(tseries)
plot(Nile)#plot the time series and assess its stationarity, variance appears to be stable across the years observed, o there is no no need for a transformation
ndiffs(Nile)#there may be a trend, which is supported by the results of the ndiffs() function

dNile <-diff(Nile)#series is differenced once(lag=1 is the default) and saved as dNile
plot(dNile)#the plot differenced looks more stationary
adf.test(dNile)#applying the ADF test to the differenced series suggest that it is now stationary


#identifying one or more reasonable models
Acf(dNile)#autocorrelation
Pacf(dNile)#partial autocorrelation

#the goal is to identify the parameters p,d and q.
#we already know that d=1 from the previous section
#get p and q by comparing the ACF and PACF plots now:

#GUidelines for selecting an ARIMA mode

#Model                    ACF             PACF
#ARIMA(p,d,0)     Trials off to zero   Zero after lag p
#ARIMA(0,d,q)     Zero after lag q     Trails off to zero
#ARIMA(p,d,q)     Trials off to zero   Trials off to zero

#this guideline is theoretical, the actual ACF and PACF may not match this exactly, but it offers a rough try 

#for the Nile time series, there appears to be one large autocorrelation at lag 1 and the partial autocorrelations trail off to zero as the lags get bigger
#this suggests trying an ARIMA(0,1,1) model

library(forecast)
fit<-arima(Nile, order=c(0,1,1))
fit
accuracy(fit)
#note that you apply the model to the original time series by specifying d=1
#it calculates first differences for you
#the coef for the moving averages(-.73) is provided along with the AIC
#if fitting other models, the AIC can help us choose which one is most reasonable
#smaller AIC suggest better models
#the accuracy measure can help determine whether the model fits with sufficient accuracy

#12. evaluating model fit
#if the model is appropriate the residuals should be normally distribuetd with the mean zero
#the autocorrelations should be zero for every possible lag
#in other words the residuals should be normally and independently distributed(no relationship between them).

qqnorm(fit$residuals)
qqline(fit$residuals)
Box.test(fit$residuals, type="Ljung-Box")
#qqnorm and qqline produce the plot. in this case, results look good
#Box.test() provides a test that the autocorrelations are all zero, the results arent significant, suggesting thatt he autocorrelations dont differ from zero
#this ARIMA model appears to fit the data well

#if the model hadnt met the assumptions of normality and zero autocorrelations, it would have been necessary to alter the model, add parameters or try a different approach
#once a fianl model ahs been chosen, it can be used to make predictions of future values

forecast(fit,3)
plot(forecast(fit,3))


#13. automated ARIMA forecasting
#ets() function to automate the selection of a best exponential model

fit<-auto.arima(sunspots)
fit
forecast(fit,3)
accuracy(fit)

#the function selects an ARIMA model with p=2, d=1, q=2 
#these are values that minimize the AIC criterion over a large number of posisble models
#the MPE and MAPE accuracy blow up because there are zero values in the series

#14. going further

#Time Series (Open University, 2006) for starting with the time series
#A Little Book of R for Time Series by Avril Coghlan(http://mng.bz/8fz0,2010) paris well with the Open University text and includes R code and examples

#forecasting: Principles and Practive(http://otexts.com/fpp,2013) highly recommeneded

#Cowpertwait&Metcalfe(2009), an excellent text on analyzing time series with R
#more advanced: Shumway & Stoffer(2010)
