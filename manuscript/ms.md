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
	\mathrm{Pr}(z_{n} = j|z_{n-1} = i, z_{n-2} = k, ..., y_{n-1},y_{n-2},...) = \mathrm{Pr}(z_{n} = j|z_{n-1}=i) = p_{ij}, \label{eq1}
\end{equation}
\end{linenomath*}
 
where the state probabilities at time point $n$ are informed by the state probabilities in the previous time points. This structure captures the inherent temporal auto-correlation of time series data in the latent state. Plainly, we expect to remain in an air or water state if that was the state in the previous time step(s). We cannot observe the true state directly so we infer $z_{n}$ by considering the probability of the observed data $y_{n}$ in either state given,

\begin{linenomath*}
\begin{equation}
	\xi_{j,n} = \mathrm{Pr}(z_{n} = j|\Omega_{n};\theta), \label{eq2}
\end{equation}
\end{linenomath*}

where the probabilities for $j = a,w$, sum to one. Our observed data is denoted by $\Omega$ and is indexed by date ($n$) while our state coefficients are captured in $\theta$ and represents the unknown model parameters that describe each state. The joint distribution of the state model $p(z_{n}|z_{n-1})$ and observation model $p(y_{n}|z_{n})$ described by,

\begin{linenomath*}
\begin{equation}
	p(z_{1:T},y_{1:T}) = [p(z_{1}) \prod_{n=2}^{T} p(z_{t}|z_{t-1})] [\prod_{t=1}^{T} p(y_{t}|z_{t})], \label{eq3}
\end{equation}
\end{linenomath*}

implies the probability of being in a given state $j$ depends on the state probabilities in the previous time step and the relative support of the data in the current time step [@Damiano:2017]. A more in depth discussion of the HMM algorithm can be found in @Hamilton:2016 and a very good tutorial has been developed by @Damiano:2017 which acted as the framework for this study.

*States: Water & Air Temperature Models*

Water and air temperatures follow the seasonal dynamics derived from changing levels of solar radiation related to the earths orbit and tilt of its axis. The shared ultimate cause of temperature and the passive exchange of energy between the two mediums, results in stream and water temperature cycles being closesly correlated. But due to differences in thermal dynamics of gases and liquids, the temperature cycles' of air and water do diverge in magnitude and timing. Therefore, we used the same model [e.g., @Shumway:2000] for both water and air temperature states described by,

\begin{linenomath*}
\begin{equation}
	y_{n} = \alpha_{z_{w,a}} + A_{z_{w\prec a}}\cos(2\pi\omega + \tau_{z_{w,a}}\pi) + \eta_{z_{w\prec a}}, \quad \eta_{z_{w\prec a}} \sim \mathcal{N}(0, f(\mathrm{S}_{n})), \label{eq4}
\end{equation}
\end{linenomath*}

where a cosine curve capturing the seasonal ocilation of temperature ($y_{n}$) is modified by state coefficients that describe the dynamics of air and water temperature ($z_{w,a}$). The mean annual temperature is captured in $\alpha$ while the range of values around $\alpha$ (i.e., amplitude) are captured in $A$. The frequency of the temperature cycle ($\omega$) is described by,

\begin{linenomath*}
\begin{equation}
	\omega = d_{n}/\gamma, \label{eq5}
\end{equation}
\end{linenomath*}

where $\gamma$ is the number of observations per cycle and $d_{n}$ the integer location of observation $n$ in the cycle. We include a seasonal adjustment in $\tau$ to shift the cosine curve from the default peak value of 1 at the origin to the point in the cycle where temperature observations begin. 

The variance around the mean temperature is a distinguishing feature between annual air and water temperature cycles. In both cases the variance is allowed to exponentially grow and decline with the mean and is centered on zero with independant scaling factors estimated in $\sigma$.

\begin{linenomath*}
\begin{equation}
	f(\mathrm{S}_{n}) = \exp(\cos(2\pi\omega + \tau_{z_{w,a}}\pi))\sigma_{z_{w\prec a}}  \label{eq6}
\end{equation}
\end{linenomath*}

We expect $A$ (eq.$\ref{eq4}$) and $\sigma$ (eq.$\ref{eq6}$) to be ordered (i.e., $z_{w}\prec z_{a}$) because the higher thermal capacity of water reduces the variance and amplitude of water's annual temperature cycle. Therefore, when the data are in the air temperature state we expect the amplitude and variance to always be larger than when data are in the water temperature state. Ordering of variables not only captures the differential dynamics of the states but increases the stability of estimation by reducing the potential for state inversion (i.e., label switching).

*Global Models*

Parameters describing air and water dynamics exhibit dependancies that scale together across locations. For instance, among locations the mean annual air and water temperatures increase together over positive values. Capturing these global relationships in a hierarchical model allows information to be shared among locations leading to better parameter estimation and subsequently greater certainty in state estimation. As mentioned, the mean annual water temperature ($\alpha_{z_{w}}$) is strongly correlated with the mean annual air temperature ($\alpha_{z_{a}}$). Here we use a linear model in log-space to describe this relationship as,

\begin{linenomath*}
\begin{equation}
	\ln(\alpha_{z_{w}}) = b_{\alpha} + m_{\alpha}\alpha_{z_{a}} + \epsilon_{\alpha}, \quad \epsilon_{\alpha} \sim \mathcal{N}(0,\varepsilon_{\alpha}), \label{eq7}
\end{equation}
\end{linenomath*}

where $b_{\alpha}$ is the mean annual water temperature when mean annual air temperature is zero and $m_{\alpha}$ describes the rate at which $\alpha_{z_{w}}$ increases with $\alpha_{z_{a}}$. Due to local characteristics of watersheds such as glaciers, ground water, canopy cover, elevation, etc., that contribute to water's temperature profile, we allowed the mean annual water temperature estimates to vary around the mean with normal errors captured in $\varepsilon_{\alpha}$. By log transforming the response variable ($\alpha_{z_{w}}$) we ensure mean annual water temperature estimates remain positive as negative values would suggest the water is frozen.

For locations where the air temperature spends a period of time below 0$\text{\textdegree}$C the water temperature is lower bound by zero. As a result $A_{z_{w}}$ reflects the mean annual water temperature (i.e., $\alpha_{z_{w}}$). We can leverage this relationship by including a model for water temperatures amplitude estimate as,

\begin{linenomath*}
\begin{equation}
	A_{z_{w}} = \alpha_{z_{w}} + \epsilon_{A}, \quad \epsilon_{A} \sim \mathcal{N}(0,\varepsilon_{A}), \label{eq7}
\end{equation}
\end{linenomath*}

where the $A_{z_{w}}$ is approximated by the mean annual water temperature estimate $\alpha_{z_{w}}$) with $\varepsilon_{A}$ capturing variation shared among sites.

Finally, the error terms for the two state models (i.e., $\sigma_{z}$ in eq. \ref{eq6}) can be constrained by assuming they are drawn from a larger population of error terms. To do this we describe the variance in these models as,

\begin{linenomath*}
\begin{equation}
	\sigma_{z} = \mu_{\sigma_{z}} + \epsilon_{\sigma_{z}}, \quad \epsilon_{\sigma{z}} \sim \mathcal{N}(0,\varepsilon_{\sigma_{z}}), \label{eq8}
\end{equation}
\end{linenomath*}

where $\mu_{\sigma_{z}}$ is the mean variance estimate with error term $\varepsilon_{\sigma_{z}}$ describing the variance around the mean. This model will draw in extreme estimates of variance thereby reducing their influence in likliehood contributions when calculating state probabilities.

*Temperature HMM*

By evaluating the likelihood of the data ($y_{n}$) given the global (eq.'s $\ref{eq6}$, $\ref{eq7}$, $\ref{eq8}$) and state models (eq.'s $\ref{eq2}$, $\ref{eq5}$), we can make inference about which state ($z_{j}$) the data are in at a given time point ($n$) by considering probabilities 



*Simulated Data*

* *Temperature*
  * *Ground*


Dependance in time series data adds additional complications to error identification in stream temperature data but can also be leveraged as a strength. Due to the seasonal periodicity of temperature, errors during certain periods may be accurate readings at other times in the cycle. As a result, treating data as independant and grouping by value will lead to frequent type 1 and type 2 errors where the data are incorrectly labeled error or accurate respectively. HMMs use the information in the previous time step to inform the subsequent time step. This characteristic of HMMs takes the autocorrelation of temperature data and leverages it to improve group estimation certainty and accuracy. 

#Results

#Discussion

#References
