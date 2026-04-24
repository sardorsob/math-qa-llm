# Assumptions

Update when the competition FAQ or your experiments contradict these.

| Area | Assumption | Risk if wrong |
|------|------------|----------------|
| Private JSONL | Each row has `id` and `question`; MCQ rows include `options`; **no** `answer` field | Loader or scorer crashes; submission missing rows |
| Submission | `response` must be the **full** model output string, not post-processed to answers only | Leaderboard extraction fails or scores unfairly |
| Free-form | Multiple `[ANS]` means multiple graded sub-answers; **all** must match | Partial credit may be zero per problem |
| MCQ | Final graded letter is extracted from the full trace (e.g. `\boxed{C}` patterns) | Format of trace must stay parseable by official eval |
| Distribution | Private split mirrors public in difficulty / domain / format mix | Strong public score may not equal private rank |
| Data license | Competition data CC BY-NC-SA 4.0 | Constraints on redistribution / commercial use |
