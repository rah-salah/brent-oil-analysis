# Assumptions & Limitations

## Assumptions
1. The Brent oil price series, once cleaned, is assumed to be free of data entry errors
   and reporting gaps beyond normal non-trading days (weekends/holidays).
2. Log returns are assumed to reasonably approximate the underlying return-generating
   process for the purposes of stationarity testing and change point detection.
3. The compiled event dataset is assumed to be a reasonably representative — though not
   exhaustive — list of major events plausibly relevant to global oil markets over the
   1987–2022 period.
4. A single (or small number of) change point(s) is assumed sufficient to capture the
   dominant structural break(s) in the series; the model does not attempt to detect every
   minor regime shift.

## Limitations

### Correlation vs. Causation (critical distinction)
Matching a statistically detected change point to the nearest date in the event dataset
demonstrates **temporal association**, not causation. Oil prices are influenced by many
simultaneous and interacting factors — supply/demand fundamentals, currency movements,
speculative trading, weather, and unrelated macroeconomic shocks — any of which could
coincide with, precede, or follow a labeled event. A change point occurring close in time
to an event is **consistent with** that event having an effect, but does not **prove** it,
and does not rule out confounding factors or coincidence. This analysis should be read as
descriptive and exploratory, not as a causal inference framework (e.g., it is not a
difference-in-differences or instrumental variable design).

### Other Limitations
- **Nearest-event matching is heuristic.** Choosing "the closest event in time" to a
  detected change point can mis-attribute a break to an unrelated event, especially when
  multiple events cluster in the same period (as in 2008 or 2020).
- **Single change point models under-fit complex histories.** Oil prices over 35 years
  have likely experienced many structural breaks; a model configured to detect only one
  or a few change points will miss others.
- **Event dataset is incomplete by construction.** No fixed list can capture every
  relevant political, economic, or supply-side development; smaller or regional events
  are likely omitted.
- **No control for confounding variables.** The model does not incorporate other
  explanatory variables (USD exchange rate, global GDP growth, inventory levels, OPEC
  spare capacity), which independently affect price and volatility.
- **Look-ahead bias risk.** Because events are known in advance and researchers are
  aware of major historical turning points, there is a risk of confirmation bias when
  selecting which events to highlight as "the" explanation for a detected break.
- **Stationarity assumption on log returns.** While ADF tests support stationarity, this
  does not guarantee constant variance (homoscedasticity) throughout, which is itself a
  motivation for the change point analysis but also a caveat on simpler downstream
  statistics computed over the full period.
