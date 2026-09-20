# Weather & daylight data-source labels

The dashboard now shows provenance for passive daylight/weather variables.

| Variable type | Dashboard source label | Meaning |
|---|---|---|
| Time in Daylight | Apple Watch | Personal daylight exposure recorded by the watch and imported through Apple Health XML. |
| Personal Daylight Exposure (% of Available Daylight) | Derived: Apple Watch + local weather | Apple Watch daylight minutes divided by locally available daylight minutes. |
| Available daylight, sunshine, cloud cover, precipitation, temperature, feels-like temperature, solar radiation | Local weather history (Open-Meteo) | Environmental conditions for the configured location; these do not prove personal exposure. |

The source appears in the Weather & Daylight table, Timing & Direction tables, personal baselines, Things to Watch cards when relevant, Associations Explorer details/dropdowns, and Data Inventory.

Screen Time remains unsupported as an automatic historical source on iPhone in this workflow and is no longer shown in the passive-context summary unless separately added later.
