# V8b Ergebnisstand

## Kurzfazit

`stateful_v2` bleibt unter robuster Multi-Delay-Selektion nur dann der staerkste Hauptpfad, wenn der Suchpfad ueber ein kleines Curriculum stabilisiert wird.

- V8a hatte gezeigt, dass harte Mittelung ueber `5,8` `stateful_v2` mechanisch stark, aber schlecht suchbar macht.
- V8b zeigt, dass ein minimales Curriculum `5 -> 5,8` genau diesen Suchpfad verbessert.
- `stateful_v2` wird damit im Kandidatenraum wieder staerker als `stateful` und knapp staerker als `stateful_plastic_hebb`.

Arbeitslesart:

- der Mechanismus von `stateful_v2` war nicht das Hauptproblem
- das Hauptproblem war die zu harte direkte Selektion
- Curriculum ist damit der richtige Verstaerker fuer den V8-Pfad

## Referenzlauf

- Label: `v8b-curriculum`
- Seeds: `7,11,13,17,19`
- Generationen: `12`
- Population: `40`
- Curriculum:
  - `phase_1`: Delay `5`
  - `phase_2`: Delays `5,8`
  - Switch: `g6`
- Varianten:
  - `stateful`
  - `stateful_v2`
  - `stateful_plastic_hebb`

Quellen:

- [v8b-curriculum.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8b-curriculum.md)
- [v8b-curriculum.candidate-features.jsonl](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8b-curriculum.candidate-features.jsonl)
- [v8b-curriculum.archive-cells.jsonl](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8b-curriculum.archive-cells.jsonl)

## Suchraum-Zusammenfassung

| variant | success_rate | mean_final_max_score | mean_score_over_delays | mean_delay_score_std |
| --- | --- | --- | --- | --- |
| stateful | 0.028750 | 2.178687 | 2.178687 | 0.045155 |
| stateful_plastic_hebb | 0.086250 | 2.248914 | 2.248914 | 0.024100 |
| stateful_v2 | 0.092917 | 2.350203 | 2.350203 | 0.043894 |

Lesart:

- `stateful_v2` ist im V8b-Suchraum die staerkste Variante
- `stateful_plastic_hebb` bleibt robuster als `stateful`, aber liegt leicht hinter `stateful_v2`
- die mittlere Delay-Streuung von `stateful_v2` sinkt gegenueber V8a sichtbar

## Vergleich gegen V8a

Fuer `stateful_v2`:

- `success_rate`: `0.055417 -> 0.092917`
- `mean_final_max_score`: `2.262508 -> 2.350203`
- `mean_delay_score_std`: `0.078632 -> 0.043894`

Lesart:

- Curriculum verbessert die Suchbarkeit von `stateful_v2` deutlich
- der Pfad produziert haeufiger starke robuste Kandidaten
- die Kandidaten werden zugleich konsistenter ueber die Delay-Familie

## Phasenbefund

| phase | candidates | success_rate | mean_score_current_phase |
| --- | --- | --- | --- |
| phase_1 | 3600 | 0.002778 | 2.071000 |
| phase_2 | 3600 | 0.135833 | 2.447536 |

Lesart:

- der Wechsel in Phase 2 fuehrt nicht zum Kollaps
- im Mittel steigt der Phasenscore nach dem Switch sogar
- Phase 2 bleibt trotzdem fordernd, weil die mittlere Elite-Qualitaet dort noch unter Phase 1 liegt

## Archivbefund

### `delay_robustness`

- `occupied_cells = 16 / 64`
- `archive_coverage = 0.250`
- `archive_mean_elite_score = 2.654772`
- perfekte robuste Elite vorhanden:
  - `score_delay_5 = 4.0`
  - `score_delay_8 = 4.0`
  - `delay_score_std = 0.0`

### `curriculum_progress`

- `phase_1`: `occupied_cells = 5`, `mean_elite_score = 2.952947`
- `phase_2`: `occupied_cells = 16`, `mean_elite_score = 2.654771`

Lesart:

- das Archiv zeigt weiterhin echte robuste Elites
- der Curriculum-Uebergang erweitert den Raum erfolgreicher Kandidaten
- die zweite Phase bleibt aber klar die haertere Suchstufe

## Empfehlung

- `stateful_v2` bleibt der Hauptmechanismuspfad
- Curriculum ist fuer robuste Multi-Delay-Selektion aktuell der richtige Hebel
- der naechste sinnvolle Schritt ist ein Boundary-Feintuning des Curriculums, nicht eine neue Zellform
