# Project Description — Volume Profile Forecast

## Problem Statement

I am interested in forecasting the trading volume profile for equities: the
percentage of the total volume traded during the regular session when the trades
are grouped in equal time bins.

Some specifics:
 + we are interested only in the regular trading session, not in the opening /
 closing auction
 + we are interested in modelling the volume profile during a regular trading
 day; we are no interested in understanding how specific events (Fed announcement,
 triple witching day, Russel rebalance) affect this profile
 + we are not focused on forecasting the absolute trading volume (as number of
 shares or notional), however we are interested in models that forecast jointly
 the volume profile and the total volume
 + we are focused on US equities, however we will consider models that were tested
 on other equity markets (EMEA, APAC) as we will test them ourselves on US stocks.
 The focus is on the methodology / model, not on the dataset used for testing.

Note that we are NOT interested in models of optimal VWAP execution.


## Literature Search

### Search Scope

Focus on classic, foundational papers and models. Look for papers cited in the field.
Ignore newer approaches based on Machine Learning (ML), AI, transformers.
I am expecting the search to generate 7-15 papers, however this is to be interpreted
only as a soft constraint.
Exclude papers that focus on VWAP execution, unless they also introduce a novel and
relevant profile forecasting method.
Use `implementable-models` research profile.

### Known Papers (optional)

None.

### Key Topics (optional)

None.


