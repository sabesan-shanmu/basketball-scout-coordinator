# Safeguard tests

21/22 cases behaved as expected

| guardrail        | case                                                      | expected_reject   | rejected   | correct   |
|:-----------------|:----------------------------------------------------------|:------------------|:-----------|:----------|
| input check      | valid matchup                                             | False             | False      | True      |
| input check      | lower-case codes                                          | False             | False      | True      |
| input check      | unknown team code                                         | True              | True       | True      |
| input check      | team scouting itself                                      | True              | True       | True      |
| input check      | instruction instead of a team                             | True              | True       | True      |
| report validator | report using only tool numbers                            | False             | False      | True      |
| report validator | invented statistic                                        | True              | True       | True      |
| report validator | number changed by rounding                                | True              | True       | True      |
| report validator | more minutes than a game lasts                            | True              | True       | True      |
| report validator | percentage above 100                                      | True              | True       | True      |
| report validator | real number, wrong meaning (known limitation)             | True              | False      | False     |
| report validator | no tools called                                           | True              | True       | True      |
| report validator | empty report                                              | True              | True       | True      |
| tool allowlist   | allowlist on: shot_chart_analyst calls top_scorers        | False             | False      | True      |
| tool allowlist   | allowlist on: shot_chart_analyst calls weakest_defenders  | True              | True       | True      |
| tool allowlist   | allowlist off: shot_chart_analyst calls top_scorers       | False             | False      | True      |
| tool allowlist   | allowlist off: shot_chart_analyst calls weakest_defenders | False             | False      | True      |
| data bounds      | real row                                                  | False             | False      | True      |
| data bounds      | player minutes above game length                          | True              | True       | True      |
| data bounds      | negative minutes                                          | True              | True       | True      |
| data bounds      | FG% above 1                                               | True              | True       | True      |
| data bounds      | negative points                                           | True              | True       | True      |
