# Benchmark Focus

Der aktuelle Hauptbeleg fuer `stateful vs stateless` ist der generationenbasierte `bit_memory`-Benchmark.

## Primaerbenchmark

- Task: `bit_memory`
- Vergleich: `stateful` vs `stateless`
- experimentelle Zusatzarme: `stateful_plastic_hebb` und spaetere AD-/Decay-Varianten
- geplanter Hauptlauf: Delay-Sweep `1,3,5,8`
- Zielmetriken:
  - `success_rate`
  - `mean_final_max_score`
  - `mean_first_success_generation`
  - `median_first_success_generation`
  - `mean_best_node_count`
  - `mean_best_enabled_connection_count`

## V5a Plasticity Exploratory Run

- Task: `bit_memory`
- Vergleich: `stateful` vs `stateless` vs `stateful_plastic` (heute: `stateful_plastic_hebb`)
- Referenzlauf:
  - Seeds `7,11,13,17,19`
  - Generationen `12`
  - Population `40`
  - Delays `1,3,5,8`
- Quelle:
  - [generation-suite-20260401T150001.286737+0000.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/generation-suite-20260401T150001.286737+0000.md)

Arbeitslesart:

- `stateful_plastic` ist im aktuellen Minimal-Hebb-Stand noch nicht konsistent besser als `stateful`.
- Das staerkste partielle Signal liegt bei `delay=5`, wo `stateful_plastic` die hoechste `success_rate` erreicht.
- Der aktuelle V5a-Pfad ist deshalb explorativ und noch kein Ersatz fuer den Hauptclaim `stateful vs stateless`.

## V5b3 Narrow-Decay Follow-up

- Task: `bit_memory`
- Vergleich: `stateful` vs `stateful_plastic_hebb` vs `stateful_plastic_ad_narrow`
- Referenzlauf:
  - Seeds `7,11,13,17,19`
  - Generationen `12`
  - Population `40`
  - Delays `5,8`
- Quelle:
  - [v5b3.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v5b3.md)

Arbeitslesart:

- `stateful_plastic_ad_narrow` bleibt auf `delay=5` und `delay=8` hinter `stateful_plastic_hebb`.
- Gegen `stateful` ist der Narrow-AD-Pfad auf beiden harten Delays ebenfalls schwaecher.
- Die Suchraumdiagnose zeigt weiterhin, dass `plastic_d` stark bei `0` haeuft und der Decay-Beitrag klein bleibt.
- `D` ist damit aktuell kein ueberzeugender Haupthebel fuer den Projektpfad.

## V6 Fast/Slow-State Follow-up

- Task: `bit_memory`
- Vergleich: `stateful` vs `stateful_v2` vs `stateful_plastic_hebb`
- Referenzlauf:
  - Seeds `7,11,13,17,19`
  - Generationen `12`
  - Population `40`
  - Delays `5,8`
- Quelle:
  - [v6.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v6.md)

Arbeitslesart:

- `stateful_v2` ist der staerkste neue Mechanismuspfad auf den harten `bit_memory`-Delays.
- Auf `delay=5` ist das Bild noch gemischt: `stateful_plastic_hebb` gewinnt die `success_rate`, `stateful_v2` aber Finalscore und Konvergenztempo.
- Auf `delay=8` schlaegt `stateful_v2` sowohl `stateful` als auch `stateful_plastic_hebb` klar.
- Die Feature-Diagnose zeigt, dass erfolgreiche `stateful_v2`-Kandidaten den Slow-State deutlich staerker nutzen.

## V7 QD-Light Archive Follow-up

- Task: `bit_memory`
- Vergleich: `stateful` vs `stateful_v2` vs `stateful_plastic_hebb`
- Archiv-Descriptoren:
  - `normalized score`
  - `slow_fast_contribution_ratio`
- Referenzlauf:
  - Seeds `7,11,13,17,19`
  - Generationen `12`
  - Population `40`
  - Delays `5,8`
- Quelle:
  - [v7.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v7.md)

Arbeitslesart:

- der reine Benchmark-Befund bleibt der bekannte V6-Stand
- neu ist hier das Archivsignal: `stateful_v2` belegt `80/128` Zellen und alle `8` Slow/Fast-Ratio-Bins
- auf `delay=8` liegen perfekte Elites sowohl in fast-dominierten als auch in stark slow-dominierten Regionen
- V7a zeigt damit, dass es mehrere funktionierende Memory-Strategien gibt und nicht nur einen globalen Champion

## V8 Robuste Multi-Delay-Selektion

- Task: `bit_memory`
- Vergleich: `stateful` vs `stateful_v2` vs `stateful_plastic_hebb`
- robuste Selektion ueber die Delay-Familie `5,8`

### V8a: harte Mittelung

- Quelle:
  - [v8a-multidelay.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8a-multidelay.md)

Arbeitslesart:

- harte Multi-Delay-Mittelung macht `stateful_v2` deutlich schwerer suchbar
- `stateful_plastic_hebb` wirkt in diesem Regime als robuster Baseline-Pfad
- der Befund spricht eher fuer einen Suchpfadfehler als fuer einen Mechanikfehler

### V8b: Curriculum `5 -> 5,8`

- Quelle:
  - [v8b.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8b.md)

Arbeitslesart:

- Curriculum verbessert die Suchbarkeit von `stateful_v2` deutlich
- `stateful_v2` wird damit im Multi-Delay-Regime wieder zum staerksten Kandidatenpfad
- Phase 2 bleibt aber eine echte Huerde

### V8c/V8d: Boundary-Policy

- Quelle:
  - [v8c.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v8c.md)

Arbeitslesart:

- `g4` ist zu frueh
- `g6` ist der konservative Standard-Schnitt
- `g8` ist fuer `stateful_v2` der staerkste Performance-Schnitt in der harten Phase
- aktuelle Policy:
  - `default_boundary = 6`
  - `performance_boundary = 8`

## Sekundaerbenchmark

- Task: `delayed_xor`
- Rolle: historischer Sanity-Check und frueher Konvergenzvergleich
- post-fix nur noch mit final-cue-Scoring und `score_ceiling=4.0` auswerten
- nicht mehr der Hauptbeleg fuer funktional relevanten internen Zustand

## Online-Ablation

- Task: `event_decision`
- Rolle: Online-/rtNEAT-artiger Zusatzbefund, nicht Primaerclaim
- aktueller Referenzlauf:
  - Seeds `7,11,13`
  - `360` Online-Steps
  - `stateful mean_final_best_score=10.667655`
  - `stateless mean_final_best_score=8.000000`
  - `run_success_rate=0.000` bei beiden Varianten

## Aktuelle Einordnung

- `bit_memory` trennt die Varianten bereits deutlich und ist deshalb der wichtigste generationenbasierte Nachweis.
- `stateful_v2` ist aktuell der interessanteste neue Upgrade-Pfad fuer internen Zustand.
- das V7a-Archiv zeigt, dass `stateful_v2` ueber mehrere unterschiedliche Memory-Strategien verfuegt.
- `stateful_plastic_hebb` bleibt ein interessanter plastischer Nebenpfad.
- der Narrow-AD-Follow-up liefert aktuell keinen Grund, `D` zum Hauptzweig zu machen.
- `delayed_xor` bleibt nuetzlich, ist aber auch nach dem Fix fuer den Kernclaim zu schwach.
- `event_decision` im Online-Modus bleibt ein zusaetzlicher Ablations-/Engineering-Befund mit Optimierungssignal, aber ohne echten Success-Durchbruch.
