# V9b Retrieval-Kalibrierung

V9b kalibriert `key_value_memory` von einer einzigen harten Retrieval-Task zu einer kleinen Stufenfamilie:

- `kv_easy`
- `kv_mid`
- `kv_full`

Zusaetzlich kommen schaerfere Retrieval-Metriken und ein zweites Retrieval-Archivprofil dazu:

- `correct_key_selected`
- `correct_value_selected`
- `query_key_match_score`
- `value_margin`
- `distractor_competition_score`
- `retrieval_mechanism`

## Technischer Stand

V9b ist als Infrastrukturpfad jetzt stabil:

- neue Taskprofile laufen ueber CLI und Config
- Candidate-Features speichern die neuen Retrieval-Diagnosen
- `retrieval_mechanism` ist als QD-Profil integriert
- die V9b-Regression gegen `bit_memory`/V8 wurde behoben

Wichtiger Fix:

- `BitMemoryEvaluator` hat kurzzeitig faelschlich `key_value_profile`-Metadaten geschrieben
- dadurch brachen V8- und `bit_memory`-Regressionstests
- der Pfad ist jetzt wieder kompatibel

## Smoke-Befunde

### `kv_easy`

Referenzlauf:

- Seeds: `7,11,13`
- Generationen: `12`
- Population: `40`

Aggregate:

| variant | success_rate | mean_final_max_score | mean_query_accuracy | mean_retrieval_score |
| --- | --- | --- | --- | --- |
| stateful | 0.000 | 3.666667 | 0.611 | 0.723 |
| stateful_plastic_hebb | 0.000 | 3.000000 | 0.500 | 0.584 |
| stateful_v2 | 0.000 | 4.666667 | 0.778 | 0.758 |

Diagnostik:

- `correct_key_selected` liegt bereits bei `1.000` fuer alle Varianten
- `stateful_v2` erreicht `correct_value_selected = 0.778`
- `value_margin` bleibt noch klein (`0.098`)

Lesart:

- Key-Selektion ist auf `kv_easy` nicht mehr das Hauptproblem
- der Engpass ist eher stabile Value-Rekonstruktion

### `kv_mid`

Kontrolllauf:

- Seeds: `7,11,13`
- Generationen: `6`
- Population: `20`

Aggregate:

| variant | success_rate | mean_final_max_score | mean_query_accuracy | mean_retrieval_score |
| --- | --- | --- | --- | --- |
| stateful | 0.000 | 5.000000 | 0.556 | 0.730 |
| stateful_plastic_hebb | 0.000 | 5.000000 | 0.556 | 0.730 |
| stateful_v2 | 0.000 | 5.666667 | 0.630 | 0.764 |

Lesart:

- `stateful_v2` fuehrt auch auf `kv_mid`
- der Vorsprung ist real, aber noch nicht gross genug fuer exakten Success

## Arbeitsurteil

- V9b ist als Analyse- und Tooling-Schritt erfolgreich
- Ticket 1 bleibt offen:
  - `kv_easy` zeigt im aktuellen Budget noch keinen echten Success
  - `kv_mid` ebenfalls noch nicht
- `stateful_v2` bleibt der staerkste Mechanismuspfad fuer weitere Kalibrierung

## Naechster sinnvoller Schritt

Nicht neue Mechanik, sondern letzter Kalibrierungsschritt:

- `kv_easy` noch leicht entschraerfen oder
- die Easy-Success-Semantik minimal pragmatischer machen

Ziel:

- mindestens eine Variante soll auf `kv_easy` sauber ueber `0` Success kippen
- danach ist `kv_mid` der naechste echte Trennbenchmark
