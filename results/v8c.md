# V8c Ergebnisstand

## Kurzfazit

Der Curriculum-Boundary-Sweep trennt jetzt sauber zwischen konservativem Standard und hartem Performance-Schnitt.

- `g4` ist zu frueh
- `g6` ist der stabile Allround-Schnitt
- `g8` ist fuer `stateful_v2` in der harten Post-Switch-Phase der staerkste Boundary

Seit V8d wird die Boundary-Entscheidung explizit auf `phase_2` ausgerichtet und nicht mehr primaer durch Gesamtmetriken ueber den ganzen Run verwaessert.

## Referenzlaeufe

- Labels:
  - `v8c-boundary4`
  - `v8c-boundary6`
  - `v8c-boundary8`
- Seeds: `7,11,13,17,19`
- Generationen: `12`
- Population: `40`
- Curriculum:
  - `phase_1`: Delay `5`
  - `phase_2`: Delays `5,8`
- Varianten:
  - `stateful`
  - `stateful_v2`
  - `stateful_plastic_hebb`

Quellen:

- [v8c-boundary4.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8c-boundary4.md)
- [v8c-boundary6.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8c-boundary6.md)
- [v8c-boundary8.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8c-boundary8.md)
- [v8c-boundary6.candidate-features.jsonl](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8c-boundary6.candidate-features.jsonl)
- [v8c-boundary8.archive-cells.jsonl](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8c-boundary8.archive-cells.jsonl)

## Overall-Vergleich

| task | delay | variant | success_rate | mean_final_max_score | mean_first_success_generation |
| --- | --- | --- | --- | --- | --- |
| bit_memory | 5->5,8@g4 | stateful | 0.600 | 3.757856 | 7.00 |
| bit_memory | 5->5,8@g4 | stateful_plastic_hebb | 0.800 | 3.691023 | 6.00 |
| bit_memory | 5->5,8@g4 | stateful_v2 | 0.600 | 3.800100 | 6.67 |
| bit_memory | 5->5,8@g6 | stateful | 0.600 | 3.788794 | 7.33 |
| bit_memory | 5->5,8@g6 | stateful_plastic_hebb | 0.800 | 3.710019 | 5.50 |
| bit_memory | 5->5,8@g6 | stateful_v2 | 0.800 | 3.799423 | 6.00 |
| bit_memory | 5->5,8@g8 | stateful | 0.400 | 3.686770 | 6.50 |
| bit_memory | 5->5,8@g8 | stateful_plastic_hebb | 1.000 | 3.702632 | 6.00 |
| bit_memory | 5->5,8@g8 | stateful_v2 | 0.800 | 3.899213 | 5.75 |

Lesart:

- `g4` ist sichtbar zu frueh fuer `stateful_v2`
- `g6` und `g8` sind overall fuer `stateful_v2` beide stark
- `g8` hat fuer `stateful_v2` den besten Finalscore und die schnellste mittlere Konvergenz

## Post-Switch-Vergleich fuer `stateful_v2`

| switch_generation | post_switch_success_rate | post_switch_mean_score_over_delays | post_switch_score_delay_8 | post_switch_delay_score_std |
| --- | --- | --- | --- | --- |
| 4 | 0.100 | 2.418441 | 2.407186 | 0.107997 |
| 6 | 0.182 | 2.538261 | 2.510284 | 0.087789 |
| 8 | 0.212 | 2.592565 | 2.565520 | 0.081791 |

Lesart:

- `g8` ist in der harten Phase durchgaengig vor `g6`
- der spaetere Wechsel verbessert fuer `stateful_v2` nicht nur die Erfolgsrate, sondern auch Delay-8-Score und Stabilitaet
- `g6` bleibt stark, ist aber in Phase 2 nicht der Leistungssieger

## Fokusduell `g6` vs `g8`

| switch_generation | overall_success_rate | overall_mean_score_over_delays | post_switch_success_rate | post_switch_mean_score_over_delays | post_switch_score_delay_8 | post_switch_delay_score_std |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.800 | 2.350203 | 0.182 | 2.538261 | 2.510284 | 0.087789 |
| 8 | 0.800 | 2.343538 | 0.212 | 2.592565 | 2.565520 | 0.081791 |

Lesart:

- overall liegen `g6` und `g8` nah beieinander
- in der harten Phase spricht das Bild aber klar fuer `g8`
- `g6` ist damit eher der konservative Schnitt, `g8` der Performance-Schnitt

## Archivbefund fuer `stateful_v2` bei `g8`

### `delay_robustness` overall

- `occupied_cells = 14 / 64`
- `archive_coverage = 0.219`
- `archive_mean_elite_score = 2.681784`

### `delay_robustness` `phase_2` only

- `occupied_cells = 11 / 64`
- `archive_coverage = 0.172`
- `archive_mean_elite_score = 2.686934`
- perfekte robuste Elite bleibt vorhanden:
  - `score_delay_5 = 4.0`
  - `score_delay_8 = 4.0`
  - `delay_score_std = 0.0`

Lesart:

- der Vorteil von `g8` ist nicht nur ein Overall-Artefakt
- auch im Phase-2-only-Archiv bleiben starke robuste Zellen sichtbar
- robuste und variablere Kandidaten werden im Archiv weiterhin sauber getrennt

## Policy

- `default_boundary = 6`
  konservativer Standard fuer gute Reproduzierbarkeit und starken Overall-Schnitt
- `performance_boundary = 8`
  bevorzugter Boundary, wenn fuer `stateful_v2` die harte Post-Switch-Leistung priorisiert wird

## Empfehlung

- `stateful_v2` bleibt der Hauptmechanismuspfad
- `g6` sollte als Default im Repo stehen bleiben
- `g8` sollte als gezielter Performance-Boundary fuer robuste Endleistung dokumentiert und weiterverfolgt werden
