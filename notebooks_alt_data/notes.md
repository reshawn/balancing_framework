Goal: To test the method on a wider variety of datasets outside of the described and intended use-case using simple tasks

Towards that variety, monash repository for a breadth of time-series datasets that are consistently formatted and available. 
For a simple task, the datasets are tailored into a binary classification problem per time-series. It is done in a way that ensures a balanced split of classes where possible and windowed rolling, cumulative, and comparative features are derived to support the values in the application of a traditional model.
The simple task formatting may not reflect the difficulty of the datasets' use-cases in practice, but allows for a scalable exploration and comparison at this stage.
Applied to more datasets, it can possibly allow for establishing a clearer definition of the attributes required in a dataset for the method to be applicable.

TODO:
x connect single instance done above to framework and eval
- loop and average to the rest in dataset, wrapping it (load and store fracdiff res in h5 file or pkl dict of dfs for whole dataset)
- debug and run on whole dataset
- repeat, cant rem how scalable eval is, number of repeats also depends on runtime for the big ones on this laptop



NOTE**********************
Frac diff function has the option of a threshold on the maximum number of dropped rows. For the sp500 (intended use-case example) it was not necessary, but a lot of these monash datasets have shorter series so its needed here. Don't want to break old code, so the change was made manually instead of moving thresh to the params.
ENDNOTE*******************


Looping:
- For each series in the dataset, run [o, o+fd, o+fod, o+fd+fod], store raw pickle results and individual run plots
Result processing:
- For each series in the dataset, comparison of 4 plot, mean min max per form
- At end of dataset, final means of [mean min max] sets

Next task:
x flatten file structure for storing raw results
X test and debug script
x run on m4, [ o, o+fd, o+fod, o+fd+fod ]
x run on single picked series
x add gluonts models to framework
- run again on single picked series
- finalize target for presentation of results i.e fields of the table
- results processing script to extract that from raw results


result table fields:
- for the same main questions around measuring consolidation of old knowledge, adaptation to new changes, and what the method does to those measures and their stabilityl; just at a different scale
per dataset, per set of forms, per consol and adapt
mean (mean min max), num of splits, mean min max series lengths, domain of dataset 
 

quick note:
look into stuff on the relationship between seasonality and stationarity
and complete the dataset info table with the domains
but so far it looks like the 'city' domain ones show more non stat series than the env ones like weather solar etc
still to check the above, but with market regimes seeming like seasons too, the nonstat seems to be about the regularity and timing of the seasons
price domains have a lot of that uncertainty, city domain would have more of it than env datasets, could maybe be a plausible reason
if it is, could allow for making up for the lack of env data fit at least in this monash collection's picture 
because then maybe the method and challenge doesnt fit so well with raw climate predictions, but the more human elements involved in the dataset task
the more that uncertainty to the seasonality might increase and therefore become a better fit
could also be entirely wrong because weather does still have some non stat series within it, but its a discussion point that could show the tie more than this subset collection alone


on gluonts model inclusion
- conditon change for splits in evaluator to split at the end instead of sklearn default
- also remembering to include the val in train for when passing train and test


quick aside on mxnet-cu112
- latest version it supports
- install driver to windows directly
- install cuda toolkit 11.2 on wsl
- it misses licudnn.8.so in cuda/lib64 so follow https://gist.github.com/sayef/8fc3791149876edc449052c3d6823d40 and download the tar from https://developer.nvidia.com/rdp/cudnn-archive
- then sudo apt install libnccl2


on gluonts model runs:
- too consuming to run full version with tuning runs at each step, running 1 fully tuned go, comparing only original vs fd/original+fd
- if that still fails, a toy example with a larger synthetic set, or the price data
- in either case, would be relying on the previous runs to make the point of the benefit to over time stability

new to do:
0 rapids frac diff? - will mention, but i dont need it right now
x run on m4 picked series with tuned transformer code
x inversion of fd function to use in place of original target
x F: programmatic tests for the other 2 conditions of using fd, verify with london and m4, then apply to rest on table for new pcts
x measure stability of the over time metrics, then run on ideally all results including the first runs on fin - still to run
x another attempt at finding a good and fitting nature dataset to run the full over time version on with a bigger model - see note below
x run PZ and plot under run3
x compare a to c again for run3 and shorter runs like the transformer ones - old transformer one has more at odds, run3 and other rf runs show both more in line, 
x diagram of eval

- run over consol evals and update accordingly
- update method flow figure for consol
- run remaining single dl runs
- plot over d comparison table with correct tick sizes
x new section in discussion on model choices and caveats
- new section in dicussion on run 3 a vs c with PZ overlay
x conclusion and future work
- get acknowledgement statements
- edit, repeating key points for emphasis, like the DL boost despite new error carried
- pick journal and format
- over time transformer runs (after submission), 10 steps to 10k rows

on stab:
- adapt generally shows less stability than the consol values, and there the transformed version regularly shows improved stability
- in consol, the values tend to be similar with or without for sp, worsened in 1 test there with the perf boost, while for the m4 where there's more shorter term value in historical values, stability there in improved, by a notable amount though half as much as the adaptation counterpart

frac diff notes:
- effective models need stationarity 
- first order differencing achieves this by subtracting the previous value from the current value. going from a series of values to a series of changes faced by those values steadies out the mean and variance over time for a cleaner and more predictable series, but it removes the historical information within how the original series values were subtly changing in each step. Trends like that of in the figure of the original series, step-changes, a shift in variance, that can lend greater predictability to the next value are seen in the original series, but because of the conflicting attributes to a model can leverage, this is lost with a first order difference for the sake of stationarity. 
- frac diff aims for a middle ground. Integer differencing where the previous value is subtracted from the last gives an operation where the options are (stationarity + loss of hist. info) or (non-stationarity + hist. info), the order of differencing, d, is 1. If the same operation were repeated on the result, d would be 2. Frac diff makes the differencing amount a continous parameter instead of an integer choice, and allows for subtracting the minimum amount that achieves stationarity, therefore retaining as much of the historical information as possible.
- 
- regular first order differencing, X_t = X_t - X_t-1, is adjusted to use a backshift operator, L or B, to define X_t more generally. Its effectively a cleaner, recursive way that allows for the parameter d to be in the general equation and for different orders of differencing to be expressed using the same equation. For example, [see screenshot fd1]. the result is an expression for the differenced series of X as (1-L)^d X_t
- adjusting that general equation to allow for d to be a real number instead of an integer is what makes the difference described above possible. Which is done by using the binomial expansion. As in, [show the generalized theorem, and where L plugs in]. Expanding with the results of L, leads to a form where a real number d can be used. Keeping in mind L^k means applying the backshift k times, so L = X_t-1, L^k = X_t-k, means the end result is our original series with decaying weights applied. In other words, the fractionally differenced series is one where each value is calculated using a decaying weighted sum of all previous values instead of only the previous value.
- In terms of using that expression for a practical implementation, of the final expansion we already have the original series and only need the weights, the binomial coefficients, to apply. Due to the recursive relationship in those coefficients, the expression can be reduced to the iterative formula used in the implementation, w_k = -w_k-1 ((d-k+1)/k) // weights_ = -weights[-1] * (diff_amt - k + 1) / k
- conditions: nonstationarity, memory or hist info (measured with: autocorr,partial autocorr,hurst?)
- cite https://link.springer.com/article/10.1007/s12572-021-00299-5 for full explanation with equations, high level desc for here))

other fd implementation notes:
- the function includes a skip step that calculates the cumulative percentage contribution of the weights and cuts off the rows below a threshold
- the second version function takes this a bit further and uses a fixed window for calculating the weights instead of a full expanding window, (something like that), is faster at the cost of some info
- rapids is another option for speed up, the adf part of the tuning doesnt benefit from it as much because thats a large number of much smaller calculations, so the added up cost of copying to and from the gpu outweighs the speed up itself
- for inverting the fd:
    - frac diff iteratively appies the most recent n weights to the n values of the series so far to produce the transformed series
    - so to reverse it, we cant reverse the entire dot product operation of the final series, but can iteratively solve for one unknown, the newest unfracdiffed value
    - for that, we subtract the dot product of the previous values from the current frac diff value, and divide by the current weight
    - i.e. if a = weights where len(weights) = n+1 and n is the length of the known original series
    - b = original_series of length n
    - c = frac diff series of length n+1 including the new to be inverted value
    - then the new value to be added to b = (c - np.dot(a[:-1], b[:-1])) / a[-1] 
    - or (the current frac diff value - the 1 less than complete frac diff calculation for this current value) / the current weight
    - and the current weight will always be 1 as the first in that series
    - NOTE: this is needed for a better comparison in the case of regression where the frac diffed series is the target being predicted, but because of how the inverted predicted values must be used in the inversion of later values within the same batch of predictions, inaccuracies would carry forward. Using the actual values for a better inversion could be possible, but for the sake of maintaining a fair forecast horizon without it, its left out.

new note on fd:
- ffd with a small window performs much better than the full series fd or fd with the skips calculated with a threshold to the pct contribution of the weights
- the former hits the high result seen before while the latter two are closer to the original value results
- start with verifying ffd again then adjusting the window from both sides, looking at the changes to the series and to performance
- other thing to fix: in the main script, the drop rows on one form doesnt reflect in the ones that dont, how does that affect the comparison? where does the drop follow through to? if an issue, reflect the biggest drop across all forms, else just align the indices better
reminder: i checked and verified that the ffd code works as intended by setting the parameters such that it gave the same result as the original function (no skips, final value)

another note but mostly for me or maybe future work sec if theres not enough there already:
- originally, the plan was to check if frac diff could allow for finding a better balance between adaptation and consolidation of models
- the assumption there being that the data shifts would lead to a choice between one and the other
- the first sets of results dont show exactly that, but there seems to be some of that holding on another glance. not the trade off in a direct way because the accuracy of fod is lowest in both adap and consol (the cost of memory-loss in this particular instance) but the stability there is better despite the lower values
- also adapt in general is higher than consol, regardless of the data form, maybe more to do with the model and data type in that case if the form is playing that much less of a role than expected
- and similarly, frac diff does find the in between of that and original, a check on a measure of the stability could show the actual trade off
- then this is a finance case in those early results, long term memory is more critical, that would play a role 
- so all in all, the reln there does matter especially with respect to the nature of the nonstat present, and theres room for more here i think
- again looking at the other forms of nonstat could lead to more room for exploring this
- so as usual the biggest room for future work would be a solid benchmark dataset for timeseries, ideally with tags on the types or properties they have to save that initial time on looking into it, because thats the main hurdle i have right now in finding datasets from other domains, i have to keep checking for these properties that should be there or might be there but is often not so simple

for the connection to climate work:
- supports lighter models
- supports new models in an interpretable manner
- niche cases across other domains can still fit the criteria
- an altered version that fits highly seasonal and mean reverting tasks instead? but with those cases, there seems to be some split differences too, like the ones with anomalous shifts versus the ones that might vary to some underlying trend that could be more fitting. in either case a step in that direction would need some more initial checks to better define the cases that fit, and considering that, a shorter step to using this for that kind of work could be in finding the specific fin use cases tied to those types of env assets and cases, which isnt in my current experience or familiarity, but some of giulia's papers seem to fit that description and could be a starting point for looking.
- datasets tried towards this:
    - all in monash of reasonable length for bigger models (some smaller non stat ones here that could be looked at for a single run small model, but wouldnt be enough to make a meaningful point that could change the current state)
    - all nature in uci
    - a few temp ones
    - river discharge and gage height - https://waterdata.usgs.gov/nwis/dv?referred_module=sw&site_no=04159130
    - fisheries and ocean canada sea level daily means - https://www.meds-sdmm.dfo-mpo.gc.ca/isdm-gdsi/twl-mne/inventory-inventaire/data-donnees-eng.asp?user=isdm-gdsi&region=CA&tst=1&no=11860


for the table on non stat
- shows a starting point of candidate series, other conditions are long term memory and absence of high seasonality
- but those tests dont cover all the cases of complex non stat series, examples m4 and london, where m4 fails a hurst exponent test but benefits from frac diff, and london where the high seasonality isnt programmatically measured easily (confirm again) and it doesnt benefit from frac diff
- so beyond the starting point that shows a meaningful presence of non stat series outside of the typical domains, a more consistent, reliable, and low-compute means of estimating the effectiveness could be useful
- further to last point, aside from stat type tests, psi/csi for measuring distribution shift in relation to a light model could also be one to check, although at a glance it seems like a proxy for checking shifts that should be detectable without considering prediction

planned runs:
- transformer reruns to save params (m4, double pred, fd in fdr)
- transformer fin
- ffn m4, m4 double pred, fin
- wavenet m4 or double pred, fin
- overtime 10 runs on fin for one of them?
- default pred len, longer context len?
- fd inversion without carrying forward the error? - should have a better result, but impractical and would be a more useful check towards better defining the situations where the method is helpful, which isnt the highest priority at this point

