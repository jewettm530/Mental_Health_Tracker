# Stoic TXT integration notes

## Source roles

- `data/raw/stoic.zip` remains the canonical structured source for question UUIDs, timestamps, contexts, and repeated observations.
- `data/raw/stoic.txt` is a complementary human-readable source for readable question/choice labels and displayed 1–5 ratings. It can also contain entries newer than the ZIP.
- Overlapping observations are not counted twice. Daily displayed ratings/text prefer TXT values; ZIP values are used as fallbacks and for observations that the TXT does not expose.

## Missing-data rule

For multi-select questions such as symptoms, triggers, and recovery methods:

- `1` = the question was answered and this option was selected.
- `0` = the question was answered and this option was not selected.
- `NULL` / `NaN` = the question was skipped or unavailable.

This prevents skipped questions from being interpreted as evidence that an option was absent. Derived Apple Health threshold flags follow the same rule when their source metric is unavailable.

## Rating scale

The TXT export displays Stoic ratings on a 1–5 scale. The structured ZIP stores these slider values internally on a 0–4 scale. The pipeline prefers TXT values and shifts ZIP fallbacks by +1 so all analysis uses the displayed 1–5 scale.

## Newly usable fields

The combined source now supports energy, stress, productivity, connectedness, subjective sleep, motivation, emotions, influences, main focus, daily plans/goals/summaries, and repeated quick mood check-ins. Long-form writing is preserved as context instead of being converted directly into simplistic sentiment scores.
