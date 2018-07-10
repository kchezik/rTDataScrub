% Cleaning Ecological Data in the Era of Big Data
% ^1,2^Kyle A. Chezik; ^1^Jonathan W. Moore
% ^1^Earth to Ocean Research Group -- Simon Fraser University 8888 University Dr. Burnaby BC, Canada V5A1S6 -- ^2^778.782.9427 -- kchezik@sfu.ca

#Abstract

#Introduction

Ecological data are often difficult and expensive to collect and despite an increasing need for broad-scale ecological studies in the face of climate change, funding for robust longitudinal and spatially extensive research is limited. In response, researchers are increasingly relying on digital remote sensing technologies to proximally capture ecological processes over large spatial extents and at higher temporal resolution than has historically been possible. For example, LiDAR is a remote sensing tool used to collect three dimensional images [e.g., @WADNR:LiDAR]. A single flight over a forest can map individual trees and be used to measure forest growth [e.g., @Caughlin:2016], canopy density [@Lee:2007], terrain topology [e.g., @WADNR:LiDAR], photosynthetic rate and even species composition [e.g., @Barbosa:2017; @Asner:2017]. Although a technology like LiDAR is currently expensive, tools such as these are often cost shared across research projects and labs, and in the future production efficiencies will reduce entry barriers. This process has already played out in many other sensor technologies like GPS, oxygen sensors, light sensors, temperature sensors and camera traps. The cost reduction and miniturization of these technologies have made mass implimentation easy and affordable. 

The quantity of data returned by remote sensing devices is both a blessing and a curse. Although, we can now collect large quantities of data over broad space and time horizons which can return refined understandings of ecosystem processes, data errors are also more likely to go unnoticed without rigorous and time consuming analysis. Although it is tempting to assume data volume will obscure the influence of erroneous data, systematic bias can lead to inaccurate conclusions which once uncovered will hurt the credability of the research or even science more generally. So while the struggle of collecting field observations may be reduced, the effort in post-processing those data is increasing. Clever post-processing tools have been developed that use a variety of rule based algorithms to eliminate many common errors. This method of error correction is referred to by Pedro Domingos as knowledge engineering [@Domingos:2015], which is wholey dependant on humans to make sense of new and unusual errors as they arise. Programing logic to identify each new source of erroneous data is laborious and often leads to complexity limitations as the number of influential variables and their combinations compound. 

Increasingly the solution to the problem of compounding complexity in big data is machine learning. Neural networks, support vector machines (SVM), principle component analysis (PCA), naive bayes, etc., are common algorithms for reducing the dimensionality of data and grouping by similarity. In so doing these methods infer underlying processes that group data without being explicity directed by a rule bound algorithm. Rather, the data themselves inform the grouping process which in theory become more defined as data volumes increase. Erroneous data could constitute it's own group but often these methods need a labeled dataset that is representative of the broader population to discern clear boundaries among groups. These broadly representative datasets are not easily compiled and are often incomplete. Tools such as PCA and Isomap can group raw data without the supervision of a labeled dataset but these tools still require humans to identify groups that constitute errors. Moreover, without considering context erroneous data can be mislabled and computational complexity often plagues these methods as data dimensionality and volume increases. To identify errors in data, which are inherently unlabeled, we need to employ unsupervised learning methods that allow the data to be probabalistically grouped under known physical constraints where certainty grows with data volume therby leveraging the 'wisdom of the crowd' [$\`{a}$ la @Surowiecki:2004]. One such method is Hidden Markov Models (HMM) where potential states are pre-defined and the probability of being in one state is estimated given the data. In this way, HMMs are at the intersection of human and machine inteligence where humans provide conceptual models from an *a priori* understanding of the processes underlying the data and then the machine bins the data under these constraints.

Water temperature governs aquatic biological processes and is a fundamental and essential component to understanding aquatic ecology. Flowing water is especially thermally dynamic and essential to understanding lotic freshwater ecosystems. Monitoring stream water temperature has become a relatively trival and routine task due to the development of cheap, long-lived, small and environmentally robust temperature sensors. As a result, sensors are being deployed at increasingly high spatial and temporal resolution, resulting in exponentially growing volumes of data even for the simplest of studies. The dynamic nature of flowing water leads to these sensors experiencing many types of error. For example, snow derived extreme spring flows in northern latitudes can lead to sensors being burried in sediment, blown onto river banks, tangled up in low hanging tree limbs or simply elevated in the water column due to shifts in the river bed. These scenarios are common, often go unidentified until retrieval and lead to mixed air, water and ground temperature data. These errors can be reduced by expert deployment but cannot altogether be avoided. Furthermore, human derived errors are also common and can be systematic or random. For instance, air temperature is often recorded during deployment and retreival of sensors which is often known and recorded. Unknown is when the sensor is found by curious passersby who remove the logger from the water column resulting in random air temperature readings of various duration. Finally, the sensors themselves can experience errors such as battery failure during extreme cold periods that become resolved during warmer periods. This is a non-exhaustive accounting of errors that may or may not be obvious in a temperature time series. Not only would it be preferable if these errors were automatically identified but a machine learning approach may also eliminate the subjectivity inherent in traditional methods of data cleaning [e.g., @Sowder:2012]. 

Here we build a Hidden Markov Model for stream temperature data with the primary goal of separating water from erroneous air temperature measurements in an effort to reduce post-processing time and human subjectivity. We uniquely build this model in a heirarchical framework therby allowing data collected at different locations and at different periods of time to contribute information to the identification of various temperature sources. This is all done in a bayesian probablistic framework that provides researchers with information as to the degree of evidence supporting a given categorization. Additionally, HMMs leverage the power of autocorrelation in time series by incorporating the level of support in temporally adjacent data to inform categorization at the current time point. Moreover, our model leverages known physical properties of seasonal temperature cycles and modeled air temperature estimates readily available to constrain the model and improve certainty and accuracy. Combined, this model provides a framework for leveraging 'big data' to identify errors in stream temperature data and automate the increasingly laborious process of cleaning these data.

#Methods

*Hidden Markov Models*

Hidden Markov Models, here-forward refered to as HMMs, are a type of state-space model that offers a probabilistic framework governing the transition between states through time [@Hamilton:2016]. In our stream temperature data we expect two states to exist, the desired state where the data exhibit stream temperature dynamics and an erroneous state where the data exhibit air temperature dynamics. The probability of being in either state is described by the first order Markov chain,

\begin{linenomath*}
\begin{equation}
	p(z_{t}|z_{t-1}), \label{eq1}
\end{equation}
\end{linenomath*}
 
where the state probabilities of air and water ($k = a,w$) at time point $t$ are informed by the state probabilities in the previous time point. This Markovian structure makes explicit our expectation of remaining in either an air or water state if that was our likely state in the previous time step. We cannot observe the true state directly so we infer $z_{t}$ by considering the probability of the observed data ($y_{t}$) in either state as,

\begin{linenomath*}
\begin{equation}
	p(y_{t}|z_{t}=k,\theta), \label{eq2}
\end{equation}
\end{linenomath*}

where coefficients describing each state and a transition matrix describing the global propensity to change between states are captured in $\theta$. The joint distribution of the state model $p(z_{t}|z_{t-1})$ and observation model $p(y_{t}|z_{t})$ described by,


\begin{linenomath*}
\begin{equation}
	p(z_{1:T},y_{1:T}) = [p(z_{1}) \prod_{t=2}^{T} p(z_{t}|z_{t-1})] [\prod_{t=1}^{T} p(y_{t}|z_{t})], \label{eq3}
\end{equation}
\end{linenomath*}

returns the full probability of being in a given state ($k = a,w$) as the relative support of the data in the current time step and the systems susceptability to a state change, weighted by the state probabilities in the previous time step [@Damiano:2017]. A more in depth discussion of the HMM model and algorithm can be found in @Hamilton:2016 and a very good tutorial that acted as the framework herein has been developed by @Damiano:2017.

*States: Water & Air Temperature Models*

Water and air temperatures follow the seasonal dynamics derived from changing levels of solar radiation related to the earths orbit and tilt of its axis. The shared ultimate cause of temperature and the passive exchange of energy between the two mediums results in stream and water temperature cycles being closesly correlated. But due to differences in thermal dynamics of gases and liquids, the temperature cycles' of air and water do diverge in magnitude and timing. Therefore, we used the same model [e.g., @Shumway:2000] for both water and air temperature states described by,

\begin{linenomath*}
\begin{equation}
	y_{t} = \alpha_{z_{w,a}} + A_{z_{w\prec a}}\cos(2\pi\omega + \tau_{z_{w,a}}\pi) + \eta_{z_{w\prec a}}, \quad \eta_{z_{w\prec a}} \sim \mathcal{N}(0, f(\mathrm{S}_{t})), \label{eq4}
\end{equation}
\end{linenomath*}

where a cosine curve capturing the seasonal ocilation of temperature ($y_{t}$) is modified by state coefficients that describe the dynamics of air and water temperature ($z_{w,a}$). The mean annual temperature is captured in $\alpha$ while the range of values around $\alpha$ (i.e., amplitude) are captured in $A$. The frequency of the temperature cycle ($\omega$) is described by,

\begin{linenomath*}
\begin{equation}
	\omega = d_{1:\gamma}/\gamma, \label{eq5}
\end{equation}
\end{linenomath*}

where $\gamma$ is the number of observations per cycle and $d_{t}$ the integer location of observation $t$ in the cycle. We include a seasonal adjustment in $\tau$ to shift the cosine curve from the default peak value of 1 at the origin to the point in the cycle where temperature observations begin. 

The variance around the mean temperature is a distinguishing feature between annual air and water temperature cycles. In both cases the variance is positive adjusted (i.e.+2), allowed to change with the mean and is centered on zero with independant scaling factors estimated in sigma ($\sigma$).

\begin{linenomath*}
\begin{equation}
	f(\mathrm{S}_{t}) = (\cos(2\pi\omega + \tau_{z_{w,a}}\pi)+2)\sigma_{z_{w\prec a}}  \label{eq6}
\end{equation}
\end{linenomath*}

We expect $A$ (eq.$\ref{eq4}$) and $\sigma$ (eq.$\ref{eq6}$) to be ordered (i.e., $z_{w}\prec z_{a}$) because the higher thermal capacity of water reduces the amplitude and variance of water's annual temperature cycle. Therefore, we expect the amplitude and variance to always be larger in the air temperature state than the water temperature state. Ordering of variables not only captures the differential dynamics of the states but increases the stability of estimation by reducing the potential for state inversion (i.e., label switching) during the multi-chain MCMC process.

*Global Models*

Parameters describing air and water dynamics exhibit dependancies that scale together. For instance, among locations the mean annual air and water temperatures increase together over positive values. Capturing these global relationships in a hierarchical model allows information to be shared among locations leading to better parameter estimation and subsequently greater certainty in state estimation. As mentioned, the mean annual water temperature ($\alpha_{z_{w}}$) is strongly correlated with the mean annual air temperature ($\alpha_{z_{a}}$) and can be decribed by a linear model in log-space as,

\begin{linenomath*}
\begin{equation}
	\ln(\alpha_{z_{w},j}) = b_{\alpha} + m_{\alpha}\alpha_{z_{a},j} + \epsilon_{\alpha}, \quad \epsilon_{\alpha} \sim \mathcal{N}(0,\varepsilon_{\alpha}), \label{eq7}
\end{equation}
\end{linenomath*}

where $b_{\alpha}$ is the mean annual water temperature when mean annual air temperature is zero and $m_{\alpha}$ describes the rate at which $\alpha_{z_{w}}$ increases with $\alpha_{z_{a}}$ among locations ($j$). Due to local characteristics of watersheds such as glaciers, ground water, canopy cover, elevation, etc., that contribute to water's temperature profile, we allowed the mean annual water temperature estimates to vary around the mean with normal errors captured in $\varepsilon_{\alpha}$. By log transforming the response variable ($\alpha_{z_{w},j}$) we ensure mean annual water temperature estimates remain positive as negative values would suggest the water is frozen.

For locations where air temperature spends a period of time below 0$\text{\textdegree}$C, liquid water temperature is lower-bound by zero. In this instance, the mean annual temperature and amplitude are interdependant such that $A_{z_{w},j}$ reflects the mean annual water temperature (i.e., $\alpha_{z_{w},j}$). We can leverage this relationship by including a model describing water temperatures amplitude as,

\begin{linenomath*}
\begin{equation}
	A_{z_{w},j} = \alpha_{z_{w},j} + \epsilon_{A}, \quad \epsilon_{A} \sim \mathcal{N}(0,\varepsilon_{A}), \label{eq8}
\end{equation}
\end{linenomath*}

where the $A_{z_{w},j}$ is approximated by the mean annual water temperature estimate ($\alpha_{z_{w},j}$) with normally distributed errors ($\varepsilon_{A}$) describing the typical variation around the mean across sites.

Finally, the error terms for the two state models (i.e., $\sigma_{z}$ in eq. \ref{eq6}) can be constrained by assuming they are drawn from two larger populations of error terms. To do this we describe the variance in these models as,

\begin{linenomath*}
\begin{equation}
	\sigma_{z,j} = \mu_{\sigma_{z}} + \epsilon_{\sigma_{z}}, \quad \epsilon_{\sigma_{z}} \sim \mathcal{N}(0,\varepsilon_{\sigma_{z}}), \label{eq9}
\end{equation}
\end{linenomath*}

where $\mu_{\sigma_{z}}$ is the mean variance estimate with error term $\varepsilon_{\sigma_{z}}$ describing the variance around the mean. These global models are particularly powerful when estimating parameters for sites where very little air or water temperature data were collected, thereby reducing overlap in the state models and improving state determination.

*Temperature HMM*

By evaluating the likelihood of the data ($y_{t}$) given the state models (eq.$\ref{eq4}$) and their governing global models (eqs. $\ref{eq7},\ref{eq8},\ref{eq9}$), weighted by the previous time step's state probabilities (eq.$\ref{eq1}$), we can make inference about which state ($k=a,w$) the data are in at each time point. Iteratively evaluating eq.$\ref{eq3}$ returns state probabilities that indicate whether the temperature data represent an air or water source as well as our certainty of that estimate. We implimented this process in the probabilistic programming language Stan [@Stan:2017]. Priors indicating our parameter value expectations were incorporated into the model thereby providing coefficient guidance to states with limited data at a given site. Prior values provided in the *Model Priors and Known Quantities* section.

*Simulated Data*

We simulated data from our state and global models to demonstrate model efficacy and test error rates under a variety of contrived but realistic scenarios. Using relationships described in Fig. 6 of @Gates:1999 and in @Pearce:1990 describing the effect of latitude ($L_{N}$) on mean annual air temperature ($\alpha_{a}$) and annual temperature range ($A_{a}$), we calculated $\alpha_{a}$ and $A_{a}$ for a variety of latitudes between 30 and 60$\text{\textdegree}$N.

\begin{linenomath*}
\begin{equation}
	\alpha_{a} = 27-(L_{N}-16)*0.65, \quad  A_{a}=L_{N}*0.4 \label{eq10}
\end{equation}
\end{linenomath*}

These $\alpha_{a}$ and $A_{a}$ values were translated to water values ($\alpha_{w}$) using the logistic function,

\begin{linenomath*}
\begin{equation}
	f(\alpha_{w}) = 27/1+\exp(-0.15*(\alpha_{a}-16)),	\label{eq11}
\end{equation}
\end{linenomath*}

where $\alpha_{w}$ values are bound between 0 and 27$\text{\textdegree}$C, had a midpoint of 16$\text{\textdegree}$C and changed over $\alpha_{a}$ values with a steepness coefficient of 0.15. We simply doubled mean annual water temperatures to calculate $A_{w}$ which ensured simulated water temperatures were rarely below 0$\text{\textdegree}$C. Uncertainty was included in all $\alpha$ and $A$ coefficients by sampling random noise from a normal distribution.

Upon random sampling a common $\tau$ estimate from a flat beta distribution for both simulated water and air data, we generated temperature data using eqs. $\ref{eq4}$, $\ref{eq5}$, and $\ref{eq6}$. Unlike in the HMM we allowed the errors in the simulated data to be correlated in order to produce data with momentum, thereby capturing the observed behaviour where temperatures above or below the mean are more likely to remain so in the subsequent day(s). In order to generate a single time series of water with erroneous air data we randomly selected chunks of water temperature data to be replaced by air temperature data. We also allowed for the first and last data point in the series to be air temperature among a random sample of sites, as air is often recorded when deploying and retrieving temperature loggers.

*Observed Data*

To demonstrate a real world application, we applied our HMM to raw stream temperature data collected at over 100 sites in the Thompson River basin in central British Columbia Canada. Collected for the purpose of understanding geomorphic impacts on stream temperature, these data are not paired with air temperature data and represent a real world example of the challenges surrounding stream temperature data cleaning. These stream temperature data were collected between July of 2014 and August of 2017 and contain examples of all aforementioned errors. Although unconfirmed, we know errors are likely due to evidence upon retrieval but the exact location of those errors in the time series is unknown. Eighteen weather stations owned and operated by Environment Canada, representing eleven distinct regions in the Thompson River basin, collected air temperature data at the same time as the water data were collected, acting as a visual reference to compare and contrast modeled source estimates with direct observations.

*Model Priors & Known Quantities*

Weakly informative priors were used for the transition state and global model parameters. We expect when temperature data are in a given state, they are more likely to remain in that state than transition to another state. Therefore, we applied beta priors with shape parameters $alpha$ = 10 and $beta$ = 2 for the probability of remaining in the current state and the inverse for the probability of transitioning between states. In the global model we expect mean annual temperatures between air and water to be positively correlated and water to be bound by zero, therefore we applied positive prior expectations on the slope ($\sim \mathcal{N}(0.1,0.5)$) and intercept ($\sim \mathcal{N}(1,1)$) parameters of eq.$\ref{eq7}$. All variance parameters in the global and state models were given broad student-t distributions to allow for extreme outlier values while also constraining the parameters to predomintely reasonable values.

Moderate to strongly informative priors were applied to some parameters where more is known about their likely values. For instance, by extracting mean annual air temperatures and associated annual temperature range estimates from the climate tool ClimateBC [@Wang:2012], we were able to provide locally adjusted (e.g., elevation, latitude), normally distributed $\alpha_{a}$ and $A_{a}$ priors with limited uncertainty ($\sigma$ = 2). During simulation these air state parameter priors were centered on the known generated values with the same uncertainty. The mean water temperature $\alpha_{w}$ and amplitude $A_{w}$ parameters were also provided moderate but less certain ($\sigma$ = 3) priors centered on the same ClimateBC estimated $\alpha_{a}$ values unless mean annual air temperatures were below 0$\text{\textdegree}$C, then we defaulted to a value of 2$\text{\textdegree}$C. Due to our knowledge of seasonal cycles we estimated $\tau$ for the air and water state assuming peak mean temperature values occur near mid-July. By evenly distributing $\tau$ values ranging from 0 to 2 between July-15^th^ to the same date of the subsequent year, we can use the date of our first temperature observation to generate a strong ($\sigma$ = 0.1) normally distributed prior.

We can directly calculate the parameter values in eq.$\ref{eq5}$ because temperature cycles occur at regular intervals. Rounding to the nearest day, there are on average 365 days in a year, thus our $\gamma$ value is some multiplication of 365 depending on the temperature sampling frequency. In this study, we averaged our temperature estimates to daily values so $\gamma$ equals 365 and each observation takes on an ordered $d$ value between 1 and 365. If we had hourly data, we would simply multiply 365 by 24 and then each observation would take on $d$ values between 1 and 8760.

*Summary Statistic Calculations*

The simulted data offer the advantage of having known and estimated source values from which we can evaluate the efficacy of our HMM. Here we calculate how often our model estimates data to be water when the true source is air (i.e., type I error), and how often our model estimates data to be air when its true source is water (i.e., type II error), both as a percentage of the number of known air or water temperature data respectively. Much like in $p$-values, these error types are subject to an arbitrary certainty boundary set by our desire to commit one type of error over the other. Our HMM requires the probability estimates between air and water states sum to 1 for each observation, thus whichever state has greater than 50% support determines the temperature source for the given observation. Alternatively, we can choose much more stringent cutoffs if we prefer to be more certain of the temperatures source. For instance, we can set our certainty at 90% before accepting data as sourced from water which would reduce type I errors but likley increase type II errors. Here we calculate the error rates as a function of several certainty boundaries to demonstrate changes in error type.

#Results

*Simulation*

Our HMM's temperature source estimation was strongly coherent with the known source in our simulated data, demonstrating an overal error rate of 0.44%. The error rate varied by site ranging as high as 1.64% with a median error rate of 0.27% and encouragingly the most common error rate was 0%. In other words, the highest error rate mislabeled 6 of the 365 days but typically only 1 day was mislabeled. 

\begin{landscape}

\begin{figure}[H]
\centering
\includegraphics[width=9in]{../images/Prob_Plot.pdf}
	\caption{Climate}
\label{fig:1}
\end{figure}

\end{landscape}

Our certainty in the source of temperature changed seasonally with strong probabilities during the summer and winter months, with weaker probabilities during the fall and spring (Figure $\ref{fig:1}$). In fact, errors exclusively occurred at the intersection of the two states (Figure $\ref{fig:2}$), during the period of seasonal transition when air and water temperatures are very similar. 

\begin{landscape}

\begin{figure}[h]
\centering
\includegraphics[width=9in]{../images/Error_Plot.pdf}
	\caption{Climate}
\label{fig:2}
\end{figure}

\end{landscape}

On average we were slightly more likely to make a type I error and label air data as water than a type II error, labeling water data as air (Figure $\ref{fig:3}$). In fact, 56% of errors were type I (*n*=18) while only 44% were type II (*n*=14). By increasing our certainty boundary for water above 50% our type I errors declined while type II errors increased exponentially (Figure $\ref{fig:3}$). By adjusting our certainty limit to 87.8% we can nearly eliminate type I errors (*n*=1), at the cost of tripling type II errors (*n*=41). Without the certainty adjstment, the absolute value type I errors were never greater than 4 and type II errors were never greater than 2 across sites. 

\begin{figure}[h]
\centering
\includegraphics[width=6.5in]{../images/ErrorType_Plot.pdf}
	\caption{Climate }
\label{fig:3}
\end{figure}

The global models were able to identify the true $\alpha_{w}$ and $A_{w}$ values among sites, as indicated by the close proximity of the data to the 1:1 line (Figure $\ref{fig:4}$).

\begin{figure}[h]
\centering
\includegraphics[width=6.5in]{../images/Global_Plot.pdf}
	\caption{Climate }
\label{fig:4}
\end{figure}

*Thompson River*

#Discussion

During the fall and spring when state models overlap, the water state was often prefered even when the known source was air, leading to a small type I error bias. 

Are errors when the states overlap in the fall and spring a concern? Discuss... 

Global Model: Overcame bias introduced by priors. Visually there was some indication that lower values may be under-estimated and larger values over-estimated. Bias towards water temperature during overlap may be because of narrower uncertainty bands or global model contributions to the water states likliehood.

Simulated data under logistic curve but the HMM uses a log transformation. Discuss the jacobian adjustement.

Higher order markov model may clean-up the mislabled data the intersection.

Making the state models a combination of sine and cosine curves may help with asymatry in the real data.

Model limited to northern sites untill the response variable in equation 7 can be logit transformed and the jacobian adjustement added to the models posterior. Also, $A_{w}$ is expected to decrease as we near the equator and leave locations where the air temperature dips below freezing for significant portions of the year. Probably good for most of North America but less useful in low latitudes and altitudes.

Limits of non-paired data when comparing air and water observations with modeled estimates.

Could allow the parameters in equation 5 to vary by year so as to allow the length of the seasonal cycle vary and account for interannual variation in the cycle length.

Could also force an order on the tau parameter because water typicallly lags behind air temperature in the seasonal cycle.

Dependance in time series data adds additional complications to error identification in stream temperature data but can also be leveraged as a strength. Due to the seasonal periodicity of temperature, errors during certain periods may be accurate readings at other times in the cycle. As a result, treating data as independant and grouping by value will lead to frequent type 1 and type 2 errors where the data are incorrectly labeled error or accurate respectively. HMMs use the information in the previous time step to inform the subsequent time step. This characteristic of HMMs takes the autocorrelation of temperature data and leverages it to improve group estimation certainty and accuracy. 

#References
