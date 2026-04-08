# NEAT Online V4

Kleines, lokal laufendes Neuroevolution-Projekt mit zwei Betriebsarten:

- generationsbasierter Legacy-Modus fuer die fruehen XOR- und Memory-Benchmarks
- neuer rtNEAT-artiger Online-Modus mit aktiver Population, Job-Queue, Rolling Metrics und Resume

Der Stack bleibt bewusst klein:

- TensorNEAT als Evolutionskern
- Python fuer numerische Evaluation, Mutation und Online-Orchestrierung
- SpacetimeDB fuer Persistenz, Lifecycle-Zustaende, Hall of Fame und Resume
- CLI statt Web-Frontend

## V4-Ueberblick

V4 verschiebt den Hauptfokus weg von harten Generationsbarrieren hin zu kontinuierlicher Evolution:

- aktive Population mit fester Groesse
- fortlaufende Evaluationsjobs
- Rolling Score pro Kandidat
- schwache Kandidaten werden laufend ersetzt
- gute Kandidaten dienen haeufiger als Eltern
- Hall of Fame bleibt separat und unveraenderlich
- Crash/Resume funktioniert auch gegen eine echte lokale SpacetimeDB

Neu als Proto-Use-Case:

- `event_memory`
  Fruehes relevantes Event, Distraktoren, spaeterer Decision-/Recall-Cue

## Architektur

### 1. TensorNEAT + Adapter

Unter [src/tensorneat](src/tensorneat) liegt ein schlank vendorter Ausschnitt des echten TensorNEAT-Kerns. Darauf sitzt eine kleine Adapter-Schicht:

- [tensorneat_adapter.py](src/evolve/tensorneat_adapter.py)
- [rtneat_scheduler.py](src/evolve/rtneat_scheduler.py)
- [candidate_registry.py](src/evolve/candidate_registry.py)

TensorNEAT liefert weiterhin Struktur, Mutation und Crossover. Der rtNEAT-artige Online-Scheduler ist eigene Python-Orchestrierung darueber, keine erfundene TensorNEAT-API.

### 2. Stateful Neuron

Das stateful Neuronmodell bleibt:

`m_t = alpha * m_(t-1) + sum(w_i * x_i) + b`

`y_t = tanh(m_t)`

Evolvierbar:

- pro Node: `bias`, `alpha`
- pro Connection: `weight`, `enabled`

Die generationenbasierte Hauptvariante bleibt weiter `stateful`. Zusaetzlich gibt es jetzt experimentelle plastische V5-Pfade:

- ein Hebb-Pfad `stateful_plastic` bzw. `stateful_plastic_hebb`
- spaetere AD-/Decay-Varianten wie `stateful_plastic_ad_narrow`
- waehrend einer Episode werden effektive Gewichte lokal angepasst
- diese Gewichtsaenderungen bleiben ephemer und werden nicht in Hall of Fame oder Genome-Blobs zurueckgeschrieben

Neu als V6-Pfad gibt es ausserdem `stateful_v2` mit zwei Zeitskalen:

- `fast state` fuer kurze reaktive Spur
- `slow state` fuer stabileren laengerfristigen Kontext
- erfolgreiche Kandidaten auf `bit_memory` nutzen den Slow-State deutlich staerker als Fehlschlaege
- dieser Pfad ist aktuell der interessanteste neue Upgrade-Kandidat fuer internen Zustand

Darauf aufbauend gibt es jetzt einen kleinen V7-QD-/Archivpfad:

- generationenbasiertes QD-light-Archiv fuer `stateful`, `stateful_v2` und `stateful_plastic_hebb`
- `mechanism_v2` fuer `stateful_v2`-interne Mechanikdiversitaet:
  - `normalized score`
  - `slow_fast_contribution_ratio`
- `general_compactness` fuer variant-uebergreifende Vergleiche:
  - `normalized score`
  - `enabled_conn_count`
- pro Descriptor-Zelle wird ein Elite-Kandidat gehalten
- der Pfad dient zuerst dem Finden, Behalten und Vergleichen verschiedener Strategien, noch nicht der aktiven Selektion

Darauf aufbauend gibt es jetzt auch einen kleinen V8-Pfad fuer robuste Selektion:

- `bit_memory` kann innerhalb eines Runs ueber mehrere Delays bewertet werden
- V8a nutzt harte Multi-Delay-Mittelung, aktuell ueber `5,8`
- V8b fuehrt ein minimales Curriculum ein:
  - Phase 1: `5`
  - Phase 2: `5,8`
- fuer robuste Kandidaten gibt es zusaetzlich die Archivprofile:
  - `delay_robustness`
  - `curriculum_progress`

Darauf aufbauend gibt es jetzt einen kleinen V9-Pfad fuer transformer-nahe Retrieval-Last:

- neue Task `key_value_memory`
- wenige relevante `store(key,value)`-Ereignisse, dazwischen Distraktoren, spaeter eine `query(key)`
- noch keine echte Attention und kein Transformer-Block
- Ziel ist selektive Retention, spaeteres Retrieval und Distraktor-Resistenz unter kleinem, kontrolliertem Kontextdruck
- neues Archivprofil `retrieval_strategy` fuer:
  - `retrieval_score`
  - `distractor_suppression_ratio`
- fuer `stateful_v2` werden zusaetzlich Store-/Query-spezifische Fast-/Slow-State-Metriken extrahiert
- V9b kalibriert diese Task jetzt weiter ueber:
  - Profile `kv_easy`, `kv_mid`, `kv_full`
  - schaerfere Retrieval-Diagnostik wie `correct_key_selected` und `value_margin`
  - das neue Archivprofil `retrieval_mechanism`

Darauf aufbauend gibt es jetzt einen kleinen V10-Pfad fuer selektiveres Gating und content-/key-value-nahe Retrieval-Mechaniken:

- neue Varianten `stateful_v2_gated`, `content_gated`, `stateful_v3_kv`
- neue, evolvierbare Content-Parameter:
  - `content_w_key`, `content_b_key`
  - `content_w_query`, `content_b_query`
  - `content_temperature`, `content_b_match`
- erweiterte Episodenmetriken fuer Gate-/Match-/Key-/Value-Ausrichtung
- neue QD-Profile:
  - `gating_mechanism`
  - `content_retrieval`
  - `kv_retrieval_mechanism`
- zusaetzliches einfaches Task-Profil `key_value_memory_trivial` fuer fruehe Sanity-Checks

Keine Spiking-Logik und keine variablen Aktivierungen. Plastizitaet bleibt aktuell ein separater Experimentalpfad, nicht der neue Default.

### 3. Online-State in SpacetimeDB

Das TS-Modul unter [spacetimedb/src/index.ts](spacetimedb/src/index.ts) enthaelt jetzt sowohl den alten Generationspfad als auch den V4-Online-Pfad.

Wichtige Online-Tabellen:

- `active_candidates`
- `evaluation_jobs`
- `evaluation_results`
- `hall_of_fame`
- `candidate_lifecycle_events`
- `online_metrics`
- `online_state`

### 4. CLI

Die CLI deckt jetzt beide Modi ab:

- `run`, `status`, `compare`, `benchmark`
- `run-online`, `status-online`, `compare-online`, `benchmark-online`

## Projektstruktur

```text
configs/
  base.yaml
  local.yaml
  delayed_xor.yaml
  bit_memory.yaml
  key_value_memory.yaml
  key_value_memory_easy.yaml
  key_value_memory_mid.yaml
  key_value_memory_full.yaml
  key_value_memory_trivial.yaml
  event_memory.yaml
  online.yaml
scripts/
  run_local.sh
  run_online.sh
  run_benchmark.sh
  simulate_crash_resume.sh
  reset_db.sh
spacetimedb/
  package.json
  tsconfig.json
  src/index.ts
src/
  main.py
  config.py
  evolve/
  tasks/
  db/
  ui/
  utils/
  tensorneat/
tests/
```

## Setup

### Python

```bash
export PATH="$HOME/.local/bin:$PATH"
python3 -m pip install --user --break-system-packages virtualenv
virtualenv "$HOME/.venvs/neat-xor-spacetimedb-v1"
source "$HOME/.venvs/neat-xor-spacetimedb-v1/bin/activate"
python -m pip install -r requirements.txt
```

### GPU unter WSL

Optional liegt [requirements-gpu.txt](requirements-gpu.txt) fuer GPU-faehiges JAX bereit:

```bash
source "$HOME/.venvs/neat-xor-spacetimedb-v1/bin/activate"
python -m pip install -r requirements-gpu.txt
```

### SpacetimeDB + Node

```bash
curl -sSf https://install.spacetimedb.com | sh -s -- -y
export PATH="$HOME/.local/bin:$PATH"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install --lts
nvm use --lts
cd spacetimedb
npm install --no-bin-links
cd ..
```

## Starten

### Generationenbasiert

```bash
PYTHONPATH=src python -m main run --store memory
```

## Benchmark-Fokus

Der aktuell wichtigste generationenbasierte Nachweis ist `bit_memory`.

- `bit_memory` ist der Primaerbenchmark fuer `stateful vs stateless`.
- plastische Varianten werden dort als experimenteller dritter Vergleichsarm mitgefuehrt.
- `delayed_xor` bleibt als historischer und sekundaerer Benchmark im Repo.
- Der Online-Pfad mit `event_decision` ist ein zusaetzlicher Ablations-/Engineering-Befund, aber nicht der Hauptclaim.

Der Grund fuer diese Gewichtung:

- `bit_memory` trennt die Varianten deutlich ueber Erfolgsrate und Finalscore.
- `stateful_v2` ist auf den harten `bit_memory`-Delays der derzeit staerkste neue Mechanismuspfad.
- das V7-Archiv zeigt jetzt zweierlei:
  - `mechanism_v2` macht sichtbar, dass `stateful_v2` mehrere unterschiedliche funktionierende Memory-Strategien hervorbringt.
  - `general_compactness` macht alle drei Varianten im Raum `Leistung x Konnektivitaet` direkt vergleichbar.
- V9a fuehrt jetzt zusaetzlich `key_value_memory` als kleinen transformer-nahen Retrieval-Benchmark ein:
  - Kontextdruck ueber relevante Stores, Distraktoren und spaetere Query
  - aktuell noch keine exakten Erfolgslaufe, aber bereits klare Unterschiede in Retrieval-Qualitaet und Archivstruktur
- V9b kalibriert diese Retrieval-Familie jetzt in kleinere Stufen:
  - `kv_easy` und `kv_mid` sind lesbarer und diagnostischer als der urspruengliche Volltask
  - die exakten Erfolgskriterien sind aber selbst dort aktuell noch nicht geknackt
- der Hebb-Pfad bleibt interessant, aber noch kein stabil besserer Ersatz fuer `stateful`.
- der spaetere AD-/Decay-Pfad ist im aktuellen V5b3-Stand schwaecher als `stateful` und `stateful_plastic_hebb`.
- `delayed_xor` liefert nach dem Scoring-Fix vor allem noch einen schwachen Konvergenz-/Sanity-Check.
- Der Hauptbeleg fuer funktional relevanten internen Zustand soll deshalb aus `bit_memory` kommen.

### Aktueller Hauptbefund

Der derzeit wichtigste Referenzlauf ist der generationenbasierte `bit_memory`-Sweep mit:

- Seeds: `7,11,13,17,19`
- Generationen: `12`
- Population: `40`
- Delays: `1,3,5,8`

Aktuelle Aggregatwerte:

| task | delay | variant | success_rate | mean_final_max_score | mean_first_success_generation |
| --- | --- | --- | --- | --- | --- |
| bit_memory | 1 | stateful | 1.000 | 3.999875 | 6.60 |
| bit_memory | 1 | stateless | 0.400 | 3.174569 | 7.00 |
| bit_memory | 3 | stateful | 1.000 | 3.963400 | 8.20 |
| bit_memory | 3 | stateless | 0.600 | 3.150180 | 7.33 |
| bit_memory | 5 | stateful | 0.800 | 3.798639 | 6.25 |
| bit_memory | 5 | stateless | 0.400 | 2.940109 | 8.00 |
| bit_memory | 8 | stateful | 0.800 | 3.678223 | 5.75 |
| bit_memory | 8 | stateless | 0.400 | 2.944937 | 8.00 |

Arbeitslesart:

- `stateful` ist auf `bit_memory` ueber alle getesteten Delays robuster als `stateless`.
- Der Vorteil zeigt sich sowohl in `success_rate` als auch in `mean_final_max_score`.
- `stateless` ist nicht chancenlos, aber systematisch schwaecher und weniger stabil.

### V5 Plasticity Experimentalpfad

Der fruehe V5a-Arm `stateful_plastic` entspricht inhaltlich dem heutigen Hebb-Pfad `stateful_plastic_hebb`.

#### V5a Hebb Exploratory Run

Fuer `bit_memory` gibt es jetzt zusaetzlich einen ersten plastischen Dreifachvergleich:

- Varianten: `stateful`, `stateless`, `stateful_plastic` (heute: `stateful_plastic_hebb`)
- Seeds: `7,11,13,17,19`
- Generationen: `12`
- Population: `40`
- Delays: `1,3,5,8`

Aktuelle Explorationswerte:

| task | delay | variant | success_rate | mean_final_max_score | mean_first_success_generation |
| --- | --- | --- | --- | --- | --- |
| bit_memory | 1 | stateful | 1.000 | 4.000000 | 6.60 |
| bit_memory | 1 | stateful_plastic | 1.000 | 3.988627 | 5.80 |
| bit_memory | 1 | stateless | 0.800 | 3.587540 | 6.00 |
| bit_memory | 3 | stateful | 0.800 | 3.800000 | 6.75 |
| bit_memory | 3 | stateful_plastic | 0.800 | 3.536077 | 6.25 |
| bit_memory | 3 | stateless | 1.000 | 3.911722 | 7.20 |
| bit_memory | 5 | stateful | 0.600 | 3.619031 | 8.00 |
| bit_memory | 5 | stateful_plastic | 1.000 | 3.578976 | 6.00 |
| bit_memory | 5 | stateless | 0.800 | 3.614798 | 9.00 |
| bit_memory | 8 | stateful | 0.800 | 3.575806 | 5.00 |
| bit_memory | 8 | stateful_plastic | 0.800 | 3.498903 | 6.00 |
| bit_memory | 8 | stateless | 0.800 | 3.308584 | 9.25 |

Arbeitslesart:

- `stateful_plastic` ist im aktuellen Minimal-Hebb-Stand noch nicht durchgaengig besser als `stateful`.
- Das staerkste partielle Signal liegt bei `delay=5`, wo `stateful_plastic` im aktuellen Lauf die hoechste `success_rate` erreicht.
- Der aktuelle V5a-Befund ist deshalb explorativ: Plastizitaet zeigt situativen Nutzen, aber noch keinen stabilen Gesamtsieg ueber den Sweep.

#### V5b3 Narrow-Decay Follow-up

Der naechste AD-/Decay-Follow-up mit enger `D`-Parametrisierung hat den AD-Pfad nicht gerettet.

- Varianten: `stateful`, `stateful_plastic_hebb`, `stateful_plastic_ad_narrow`
- Seeds: `7,11,13,17,19`
- Generationen: `12`
- Population: `40`
- Delays: `5,8`

| task | delay | variant | success_rate | mean_final_max_score | mean_first_success_generation |
| --- | --- | --- | --- | --- | --- |
| bit_memory | 5 | stateful | 0.600 | 3.619031 | 8.00 |
| bit_memory | 5 | stateful_plastic_hebb | 0.800 | 3.291882 | 8.25 |
| bit_memory | 5 | stateful_plastic_ad_narrow | 0.400 | 3.122112 | 7.50 |
| bit_memory | 8 | stateful | 0.800 | 3.575806 | 5.00 |
| bit_memory | 8 | stateful_plastic_hebb | 0.600 | 3.276288 | 5.00 |
| bit_memory | 8 | stateful_plastic_ad_narrow | 0.200 | 2.667075 | 7.00 |

Arbeitslesart:

- `stateful_plastic_ad_narrow` bleibt auf den harten Delays hinter `stateful_plastic_hebb` und `stateful`.
- Die Feature-Analyse zeigt weiter, dass `plastic_d` stark bei `0` haeuft und der Decay-Beitrag klein bleibt.
- Die enge `D`-Parametrisierung verbessert die Suchbarkeit nicht genug, um den AD-Pfad aktuell attraktiv zu machen.
- Fuer den Projektstand sollte `D` deshalb nicht als Haupthebel gelesen werden.

Quelle:

- [v5b3.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v5b3.md)

### V6 Fast/Slow-State Follow-up

Der erste staerkere neue Mechanismuspfad nach den Plasticity-Experimenten ist `stateful_v2`.

- Varianten: `stateful`, `stateful_v2`, `stateful_plastic_hebb`
- Seeds: `7,11,13,17,19`
- Generationen: `12`
- Population: `40`
- Delays: `5,8`

| task | delay | variant | success_rate | mean_final_max_score | mean_first_success_generation |
| --- | --- | --- | --- | --- | --- |
| bit_memory | 5 | stateful | 0.600 | 3.619031 | 8.00 |
| bit_memory | 5 | stateful_plastic_hebb | 1.000 | 3.578976 | 6.00 |
| bit_memory | 5 | stateful_v2 | 0.800 | 3.800832 | 5.75 |
| bit_memory | 8 | stateful | 0.800 | 3.575806 | 5.00 |
| bit_memory | 8 | stateful_plastic_hebb | 0.800 | 3.498903 | 6.00 |
| bit_memory | 8 | stateful_v2 | 1.000 | 3.990067 | 4.80 |

Arbeitslesart:

- `stateful_v2` ist auf `delay=8` der klare Gewinner ueber Erfolgsrate, Finalscore und Konvergenztempo.
- Auf `delay=5` bleibt das Bild gemischt: `stateful_plastic_hebb` gewinnt die Erfolgsrate, `stateful_v2` aber Finalscore und mittlere Zeit bis zum ersten Erfolg.
- Die Suchraumdiagnose zeigt, dass erfolgreiche `stateful_v2`-Kandidaten den Slow-State deutlich staerker nutzen als Fehlschlaege.
- Der naechste Hauptpfad des Projekts sollte deshalb eher `stateful_v2` als weitere AD-/Decay-Varianten sein.

Quelle:

- [v6.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v6.md)

### V7 QD-Light Archive Follow-up

Der neue V7-Pfad erweitert `stateful_v2` nicht ueber neue Mechanik, sondern ueber eine neue Sicht auf den Suchraum.

- Varianten: `stateful`, `stateful_v2`, `stateful_plastic_hebb`
- Seeds: `7,11,13,17,19`
- Generationen: `12`
- Population: `40`
- Delays: `5,8`
- Archiv-Descriptor-Profile:
  - `mechanism_v2`
    - `normalized score`
    - `slow_fast_contribution_ratio`
  - `general_compactness`
    - `normalized score`
    - `enabled_conn_count`

Die Benchmark-Tabelle bleibt dabei der bekannte V6-Befund:

| task | delay | variant | success_rate | mean_final_max_score | mean_first_success_generation |
| --- | --- | --- | --- | --- | --- |
| bit_memory | 5 | stateful | 0.600 | 3.619031 | 8.00 |
| bit_memory | 5 | stateful_plastic_hebb | 1.000 | 3.578976 | 6.00 |
| bit_memory | 5 | stateful_v2 | 0.800 | 3.800832 | 5.75 |
| bit_memory | 8 | stateful | 0.800 | 3.575806 | 5.00 |
| bit_memory | 8 | stateful_plastic_hebb | 0.800 | 3.498903 | 6.00 |
| bit_memory | 8 | stateful_v2 | 1.000 | 3.990067 | 4.80 |

Der eigentliche neue Befund liegt im Archiv, und er ist jetzt zweigeteilt.

#### V7a: `mechanism_v2`

Das mechanism-orientierte Profil trennt `stateful_v2` klar von den Vergleichsvarianten:

- `stateful`: `occupied_cells = 14 / 128`, `archive_coverage = 0.109`, `strategy_diversity = 1`, `archive_mean_elite_score = 2.375265`
- `stateful_plastic_hebb`: `occupied_cells = 14 / 128`, `archive_coverage = 0.109`, `strategy_diversity = 1`, `archive_mean_elite_score = 2.346970`
- `stateful_v2`: `occupied_cells = 80 / 128`, `archive_coverage = 0.625`, `strategy_diversity = 8`, `archive_mean_elite_score = 2.894709`

Wichtige Lesart:

- `stateful_v2` hat nicht nur einen guten Champion, sondern mehrere funktionierende Memory-Strategien
- auf `delay=8` liegen perfekte Elites sowohl in fast-dominierten als auch in stark slow-dominierten Regionen
- `stateful` und `stateful_plastic_hebb` bleiben in diesem Profil mechanisch weitgehend eindimensional

#### V7b: `general_compactness`

Das variant-uebergreifende Profil zeigt einen anderen Aspekt des Suchraums:

- `stateful`: `occupied_cells = 62 / 128`, `archive_coverage = 0.484`, `strategy_diversity = 6`, `archive_mean_elite_score = 2.521583`
- `stateful_plastic_hebb`: `occupied_cells = 62 / 128`, `archive_coverage = 0.484`, `strategy_diversity = 7`, `archive_mean_elite_score = 2.726169`
- `stateful_v2`: `occupied_cells = 56 / 128`, `archive_coverage = 0.438`, `strategy_diversity = 6`, `archive_mean_elite_score = 2.863462`

Wichtige Lesart:

- im Raum `Leistung x Konnektivitaet` sind alle drei Varianten breit genug, also nicht mehr degeneriert
- `stateful_v2` ist dort nicht der breiteste Suchraum, hat aber die beste mittlere Elite-Qualitaet
- `stateful_plastic_hebb` wirkt im Kompaktheitsraum diverser als im reinen Fast/Slow-Mechanikprofil

Gesamtlesart:

- `stateful_v2` eroefnet einen echten neuen Mechanikraum
- `general_compactness` bestaetigt, dass die Vergleichsvarianten durchaus Strukturdiversitaet haben
- `stateful_v2` kombiniert aktuell die staerkste Elite-Qualitaet mit der reichsten sichtbaren Mechanikdiversitaet

Quelle:

- [v7.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v7.md)

### V8 Robuste Multi-Delay-Selektion

V8 verschiebt die Frage von "welcher Kandidat loest einen festen Delay?" zu "welcher Kandidat bleibt ueber eine kleine Delay-Familie stabil?".

- Varianten: `stateful`, `stateful_v2`, `stateful_plastic_hebb`
- Seeds: `7,11,13,17,19`
- Generationen: `12`
- Population: `40`
- harte Multi-Delay-Selektion ueber `5,8`

#### V8a: harte Mittelung ueber `5,8`

| task | delay | variant | success_rate | mean_final_max_score | mean_first_success_generation |
| --- | --- | --- | --- | --- | --- |
| bit_memory | 5,8 | stateful | 0.600 | 3.790500 | 5.67 |
| bit_memory | 5,8 | stateful_plastic_hebb | 0.800 | 3.792447 | 5.25 |
| bit_memory | 5,8 | stateful_v2 | 0.400 | 3.578509 | 4.50 |

Lesart:

- unter harter Multi-Delay-Mittelung bleibt `stateful_v2` mechanisch stark, wird aber deutlich schwerer suchbar
- `stateful_plastic_hebb` wirkt in diesem Regime als robuster und leichter auffindbarer Pfad
- der Befund spricht eher gegen einen Mechanikfehler als gegen einen Suchpfadfehler

#### V8b: Curriculum `5 -> 5,8`

Fuer V8b wird die Multi-Delay-Selektion nicht sofort hart gefahren:

- Generationen `0-5`: nur Delay `5`
- Generationen `6-11`: Delays `5,8`

Der volle V8b-Hauptlauf zeigt im Suchraum:

- `stateful`: `success_rate = 0.028750`, `mean_final_max_score = 2.178687`
- `stateful_plastic_hebb`: `success_rate = 0.086250`, `mean_final_max_score = 2.248914`
- `stateful_v2`: `success_rate = 0.092917`, `mean_final_max_score = 2.350203`

Die wichtigste Vergleichslesart fuer `stateful_v2` gegen V8a:

- `success_rate`: `0.055417 -> 0.092917`
- `mean_final_max_score`: `2.262508 -> 2.350203`
- `mean_delay_score_std`: `0.078632 -> 0.043894`

Archiv- und Phasenlesart:

- `delay_robustness`: `occupied_cells = 16 / 64`, `archive_coverage = 0.250`, `archive_mean_elite_score = 2.654772`
- perfekte robuste Elite bleibt sichtbar:
  - `score_delay_5 = 4.0`
  - `score_delay_8 = 4.0`
  - `delay_score_std = 0.0`
- `curriculum_progress` zeigt:
  - `phase_1`: `occupied_cells = 5`, `mean_elite_score = 2.952947`
  - `phase_2`: `occupied_cells = 16`, `mean_elite_score = 2.654771`

Gesamtlesart:

- V8a zeigt, dass `stateful_v2` unter robuster Selektion nicht automatisch der beste Suchpfad ist
- V8b zeigt, dass Curriculum genau diesen Suchpfad verbessert
- `stateful_v2` wird mit Curriculum wieder zum staerksten Kandidatenpfad auf Kandidatenebene
- Phase 2 bleibt trotzdem eine echte Huerde; der Uebergang ist besser, aber noch nicht "gratis"

Arbeitsurteil:

- `stateful_v2` bleibt der Hauptmechanismus
- Curriculum ist aktuell der richtige Verstaerker fuer robuste Multi-Delay-Selektion
- der naechste plausible Schritt ist eher V8c-Feintuning des Uebergangs als ein neuer Mechanismus

#### V8c: Boundary-Sweep `g4 / g6 / g8`

V8c haelt das Curriculum-Schema konstant und sweeped nur den Wechselpunkt:

- `g4`: frueher Wechsel in die harte Phase
- `g6`: mittlerer Wechselpunkt, entspricht dem bisherigen V8b-Default
- `g8`: spaeter Wechsel, laengerer Aufbau der Memory-Mechanik vor der harten Robustheitsphase

Boundary-Referenzlaeufe:

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

V8d richtet die Boundary-Entscheidung jetzt explizit auf die harte `phase_2` aus, statt Gesamtmetriken ueber den ganzen Run zu priorisieren.

Post-switch-Lesart fuer `stateful_v2`:

- `g4`: `post_switch_success_rate = 0.100`, `post_switch_mean_score_over_delays = 2.418441`
- `g6`: `post_switch_success_rate = 0.182`, `post_switch_mean_score_over_delays = 2.538261`
- `g8`: `post_switch_success_rate = 0.212`, `post_switch_mean_score_over_delays = 2.592565`

Arbeitsurteil:

- `g4` ist zu frueh
- `g6` ist der konservative, gut reproduzierbare Allround-Schnitt
- `g8` ist fuer `stateful_v2` der aktuell staerkste Performance-Boundary in der harten Phase

Aktuelle Policy:

- `default_boundary = 6`
- `performance_boundary = 8`

### V9a Transformer-nahe Retrieval-Task

V9a fuehrt mit `key_value_memory` eine kleine, kontrollierte Retrieval-Taskfamilie ein:

- relevante `store(key,value)`-Schritte
- Distraktoren zwischen Speicherung und Abruf
- spaetere `query(key)` mit Zielwert-Rekonstruktion
- Curriculum aktuell als `3 -> 8 @ g6`

Die Task soll noch keinen Transformer simulieren, aber bereits die fuer sequenzielles Retrieval wichtigen Eigenschaften messen:

- selektive Retention
- spaeteres Retrieval
- Resistenz gegen Distraktoren
- mechanisch interpretierbare Abrufpfade

#### V9a: Curriculum `3 -> 8`

Vollstaendiger Referenzlauf:

- Task: `key_value_memory`
- Varianten: `stateful`, `stateful_v2`, `stateful_plastic_hebb`
- Seeds: `7,11,13,17,19`
- Generationen: `12`
- Population: `40`
- Curriculum:
  - Phase 1: `3`
  - Phase 2: `8`
  - Switch: `g6`

Benchmark-Aggregate:

| task | delay | variant | success_rate | mean_final_max_score | mean_query_accuracy | mean_retrieval_score |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3->8@g6 | stateful | 0.000 | 6.200000 | 0.517 | 0.754 |
| key_value_memory | 3->8@g6 | stateful_plastic_hebb | 0.000 | 6.400000 | 0.533 | 0.767 |
| key_value_memory | 3->8@g6 | stateful_v2 | 0.000 | 7.400000 | 0.617 | 0.789 |

Phase-2-Suchraumlesart:

- `stateful`: `mean_final_max_score = 4.434167`, `query_accuracy = 0.369583`, `retrieval_score = 0.674603`
- `stateful_plastic_hebb`: `mean_final_max_score = 4.242500`, `query_accuracy = 0.353542`, `retrieval_score = 0.665772`
- `stateful_v2`: `mean_final_max_score = 4.134167`, `query_accuracy = 0.344444`, `retrieval_score = 0.638897`

Archiv- und Mechaniklesart:

- `stateful_v2` im Profil `retrieval_strategy`:
  - overall: `occupied_cells = 37 / 64`, `archive_mean_elite_score = 4.000000`, `best_score = 8.0`
  - `phase_2`: `occupied_cells = 27 / 64`, `archive_mean_elite_score = 4.148148`, `best_score = 8.0`
- `stateful`: `occupied_cells = 36 / 64`, `archive_mean_elite_score = 3.916667`, `best_score = 7.0`
- `stateful_plastic_hebb`: `occupied_cells = 37 / 64`, `archive_mean_elite_score = 3.864865`, `best_score = 8.0`
- nur `stateful_v2` zeigt dabei systematisch `slow_query_coupling > 0`

Gesamtlesart:

- V9a ist technisch gruen und die neue Task trennt die Varianten sinnvoll
- es gibt im aktuellen Budget noch keine exakten Erfolge; die Task bleibt also hart genug
- `stateful_v2` ist im Vollbenchmark der staerkste Hauptpfad fuer Retrieval-Qualitaet unter Distraktordruck
- in `phase_2` sind die Baselines im Populationsmittel noch konkurrenzfaehig, aber `stateful_v2` hat die staerksten Elite-/Archivsignale
- der Abrufpfad von `stateful_v2` wirkt mechanistisch getragen, nicht nur reaktiv

Arbeitsurteil:

- `key_value_memory` bleibt vorerst ein zweiter Hauptbenchmark neben `bit_memory`
- `stateful_v2` ist auch auf der neuen Retrieval-Aufgabe der interessanteste Mechanismuspfad
- der naechste Schritt sollte eher mehr Budget / feinere Curriculum-Arbeit sein als eine neue Modellfamilie

### V9b Retrieval-Kalibrierung

V9b macht aus `key_value_memory` eine besser lesbare, aber weiterhin kleine Retrieval-Familie:

- Task-Profile:
  - `kv_easy`
  - `kv_mid`
  - `kv_full`
- neue Retrieval-Diagnostik:
  - `correct_key_selected`
  - `correct_value_selected`
  - `query_key_match_score`
  - `value_margin`
  - `distractor_competition_score`
- neue mechanistische Features:
  - `retrieval_state_alignment`
  - `slow_query_coupling`
  - `store_query_state_gap`
- neues Archivprofil:
  - `retrieval_mechanism`

Aktueller V9b-Stand:

- technisch ist der Pfad jetzt stabil:
  - neue Task-Profile, CLI-Flags, Feature-Extraktion und Archivprofile laufen
  - die Regression gegen `bit_memory`/V8 wurde behoben
- inhaltlich ist die Kalibrierung noch nicht fertig:
  - `kv_easy` zeigt im aktuellen Budget noch keine exakten Erfolge
  - `kv_mid` ebenfalls noch nicht
  - `stateful_v2` liegt dabei aber schon konsistent vorne

Aktuelle Smoke-Lesart:

- `kv_easy` mit 12 Generationen / Population 40:
  - `stateful_v2`: `mean_final_max_score = 4.666667`, `mean_query_accuracy = 0.778`
  - `stateful`: `3.666667`, `0.611`
  - `stateful_plastic_hebb`: `3.000000`, `0.500`
- `kv_mid` mit 6 Generationen / Population 20:
  - `stateful_v2`: `mean_final_max_score = 5.666667`, `mean_query_accuracy = 0.630`
  - `stateful`: `5.000000`, `0.556`
  - `stateful_plastic_hebb`: `5.000000`, `0.556`

Der derzeitige Engpass ist nicht mehr primaer die Key-Selektion, sondern eher die stabile Value-Rekonstruktion:

- auf `kv_easy` liegt `correct_key_selected` bereits bei `1.000` fuer alle Varianten
- `stateful_v2` hat dort aber nur `correct_value_selected = 0.778`
- `value_margin` bleibt noch klein; die Task ist also lesbarer, aber noch nicht sauber geloest

Arbeitsurteil:

- V9b ist als Infrastruktur- und Analyse-Upgrade erfolgreich
- Ticket 1 ist inhaltlich noch offen: `kv_easy` muss noch in einen Bereich kippen, in dem wenigstens eine Variante echten Success zeigt
- `stateful_v2` bleibt der staerkste Mechanismuspfad fuer die weitere Kalibrierung

Quelle:

- [v9a-curriculum.md](/mnt/c/Users/joach/NEAT.wsl.projekt/results/v9a-curriculum.md)

### Delayed XOR Einordnung

`delayed_xor` bleibt im Projekt, wird aktuell aber nur noch als sekundaerer Benchmark gelesen.

- Das fruehere `score_ceiling=16.0` war ein Evaluationsfehler; `delayed_xor` scoret jetzt korrekt nur noch den finalen Cue.
- Im post-fix Referenzlauf mit Seeds `7,11,13,17,19` liegt `stateful` bei `mean_final_max_score=2.985422`, `stateless` bei `2.553476`.
- Die Variante trennt dort nur leicht und liefert unter dem aktuellen Budget keine Erfolgslaeufe.
- Der Lauf ist weiterhin nuetzlich als historischer Sanity-Check.
- Der Kernclaim zu funktional relevantem internem Zustand soll sich nicht auf `delayed_xor` stuetzen.

Wichtig:

- Alte `delayed_xor`-Exports vor dem Fix mit `score_ceiling=16.0` sind inhaltlich ueberholt.
- Fuer aktuelle Aussagen sollte nur der post-fix Lauf herangezogen werden.

### Online-Einordnung

Der Online-Pfad mit `event_decision` bleibt aktuell ein ergaenzender Ablations- und Engineering-Befund, nicht der Primaerbeleg.

Referenzlauf:

- Seeds: `7,11,13`
- Online-Steps: `360`
- Task: `event_decision`

Aktuelle Online-Aggregate:

| task | steps | variant | run_success_rate | mean_final_best_score | mean_final_rolling_avg_score | mean_hall_of_fame_growth |
| --- | --- | --- | --- | --- | --- | --- |
| event_decision | 360 | stateful | 0.000 | 10.667655 | 7.088538 | 6.67 |
| event_decision | 360 | stateless | 0.000 | 8.000000 | 6.800000 | 1.00 |

Arbeitslesart:

- `stateful` zeigt im Online-Modus einen stabilen Optimierungs- und Suchvorteil.
- Unter dem aktuellen Budget gibt es aber noch keinen echten Success-Durchbruch.
- Der Online-Pfad bleibt damit ein interessanter Zusatzbefund, aber nicht der staerkste inhaltliche Claim des Projekts.

### Aktuelle Prioritaet

Die naechsten Schritte liegen jetzt nicht in neuen Features, sondern in belastbarer Auswertung und Dokumentation.

- `bit_memory` bleibt der Primaerbenchmark.
- `stateful_v2` ist der neue Haupt-Experimentalpfad fuer bessere interne Speichermechanismen.
- der V7-Archivpfad ist der neue Hauptpfad fuer Suchraumverstaendnis und spaetere QD-Selektion.
- `mechanism_v2` bleibt das beste Profil fuer echte `stateful_v2`-Mechanikdiversitaet.
- `general_compactness` ist jetzt das einfache variant-uebergreifende Vergleichsprofil.
- `delay_robustness` und `curriculum_progress` sind jetzt die relevanten V8-Profile fuer robuste Kandidaten.
- `key_value_memory` ist jetzt der kleine transformer-nahe Retrieval-Benchmark fuer selektive Retention und spaeteren Abruf.
- `retrieval_strategy` ist das passende V9a-Archivprofil fuer Retrieval-Mechanik unter Distraktordruck.
- `stateful_plastic_hebb` bleibt ein starker robuster Baseline-Pfad, vor allem ohne Curriculum.
- V8b-Curriculum macht `stateful_v2` im robusten Multi-Delay-Regime wieder besser suchbar.
- V8c/V8d trennen jetzt sauber zwischen konservativem Standard und hartem Performance-Schnitt:
  - `default_boundary = 6`
  - `performance_boundary = 8`
- V9a zeigt bereits einen Vorsprung von `stateful_v2` bei Retrieval-Qualitaet, aber noch keine voll geloeste neue Benchmark-Familie.
- der AD-/Decay-Pfad bleibt im aktuellen Stand nachrangig.
- `delayed_xor` bleibt ein Nebenbefund.
- `event_decision` online bleibt ein ergaenzender Ablationsbefund.
- Neuer Feature-Scope wird bewusst klein gehalten, bis die Kernmessungen sauber stehen.

### Generationenbasierte Benchmark-Suite

Fuer den generationenbasierten `stateful vs stateless`-Nachweis gibt es einen Batch-Pfad mit Export nach `results/`.

Empfohlener Hauptlauf fuer den Primaerbenchmark:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks bit_memory \
  --task-delay-sweep bit_memory=1,3,5,8 \
  --seeds 7,11,13,17,19 \
  --output-dir results
```

Explorativer V5a-Dreifachvergleich mit Plastizitaet:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks bit_memory \
  --variants stateful,stateless,stateful_plastic_hebb \
  --task-delay-sweep bit_memory=1,3,5,8 \
  --seeds 7,11,13,17,19 \
  --output-dir results
```

Harter V5b3-Follow-up fuer die AD-Variante:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks bit_memory \
  --variants stateful,stateful_plastic_hebb,stateful_plastic_ad_narrow \
  --task-delay-sweep bit_memory=5,8 \
  --seeds 7,11,13,17,19 \
  --output-dir results
```

V6-Follow-up fuer die Zwei-Zeitskalen-State-Variante:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks bit_memory \
  --variants stateful,stateful_v2,stateful_plastic_hebb \
  --task-delay-sweep bit_memory=5,8 \
  --seeds 7,11,13,17,19 \
  --output-dir results
```

V7-QD-light-Archiv mit beiden Profilen:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks bit_memory \
  --variants stateful,stateful_v2,stateful_plastic_hebb \
  --task-delay-sweep bit_memory=5,8 \
  --seeds 7,11,13,17,19 \
  --output-dir results \
  --label v7b-qdlight
```

Archiv-Report fuer den `stateful_v2`-Mechanikraum:

```bash
PYTHONPATH=src python -m main analyze-archive \
  --store memory \
  --benchmark-label v7b-qdlight \
  --task bit_memory \
  --variant stateful_v2 \
  --qd-profile mechanism_v2 \
  --output-dir results
```

Archiv-Report fuer den variant-uebergreifenden Kompaktheitsraum:

```bash
PYTHONPATH=src python -m main analyze-archive \
  --store memory \
  --benchmark-label v7b-qdlight \
  --task bit_memory \
  --variant stateful_v2 \
  --qd-profile general_compactness \
  --output-dir results
```

V8a-Hauptlauf fuer harte Multi-Delay-Selektion:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks bit_memory \
  --variants stateful,stateful_v2,stateful_plastic_hebb \
  --seeds 7,11,13,17,19 \
  --config configs/base.yaml \
  --config configs/bit_memory.yaml \
  --config configs/local.yaml \
  --evaluation-delay-steps 5,8 \
  --temporal-delay-steps 8 \
  --output-dir results \
  --label v8a-multidelay
```

V8b-Hauptlauf mit minimalem Curriculum:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks bit_memory \
  --variants stateful,stateful_v2,stateful_plastic_hebb \
  --seeds 7,11,13,17,19 \
  --config configs/base.yaml \
  --config configs/bit_memory.yaml \
  --config configs/local.yaml \
  --curriculum-enabled \
  --curriculum-phase-1-delays 5 \
  --curriculum-phase-2-delays 5,8 \
  --curriculum-phase-switch-generation 6 \
  --temporal-delay-steps 8 \
  --output-dir results \
  --label v8b-curriculum
```

V8b-Suchraumreport:

```bash
PYTHONPATH=src python -m main analyze-search-space \
  --store memory \
  --benchmark-label v8b-curriculum \
  --task bit_memory \
  --output-dir results
```

V8b-Archivreport fuer robuste Kandidaten:

```bash
PYTHONPATH=src python -m main analyze-archive \
  --store memory \
  --benchmark-label v8b-curriculum \
  --task bit_memory \
  --variant stateful_v2 \
  --qd-profile delay_robustness \
  --output-dir results
```

V8b-Archivreport fuer den Curriculum-Uebergang:

```bash
PYTHONPATH=src python -m main analyze-archive \
  --store memory \
  --benchmark-label v8b-curriculum \
  --task bit_memory \
  --variant stateful_v2 \
  --qd-profile curriculum_progress \
  --output-dir results
```

V8c-Boundary-Sweep:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks bit_memory \
  --variants stateful,stateful_v2,stateful_plastic_hebb \
  --seeds 7,11,13,17,19 \
  --config configs/base.yaml \
  --config configs/bit_memory.yaml \
  --config configs/local.yaml \
  --curriculum-enabled \
  --curriculum-phase-1-delays 5 \
  --curriculum-phase-2-delays 5,8 \
  --curriculum-phase-switch-generation 6 \
  --temporal-delay-steps 8 \
  --output-dir results \
  --label v8c-boundary6
```

V8d-Report fuer die Boundary-Entscheidung:

```bash
PYTHONPATH=src python -m main analyze-curriculum-boundaries \
  --benchmark-labels v8c-boundary4,v8c-boundary6,v8c-boundary8 \
  --task bit_memory \
  --output-dir results
```

Phase-2-only Suchraumreport:

```bash
PYTHONPATH=src python -m main analyze-search-space \
  --store memory \
  --benchmark-label v8c-boundary6 \
  --task bit_memory \
  --curriculum-phase phase_2 \
  --output-dir results
```

Phase-2-only Archivreport:

```bash
PYTHONPATH=src python -m main analyze-archive \
  --store memory \
  --benchmark-label v8c-boundary8 \
  --task bit_memory \
  --variant stateful_v2 \
  --qd-profile delay_robustness \
  --curriculum-phase phase_2 \
  --output-dir results
```

V9a-Curriculum-Hauptlauf fuer transformer-nahes Retrieval:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks key_value_memory \
  --variants stateful,stateful_v2,stateful_plastic_hebb \
  --seeds 7,11,13,17,19 \
  --config configs/base.yaml \
  --config configs/local.yaml \
  --curriculum-enabled \
  --output-dir results \
  --label v9a-curriculum
```

V9a-Suchraumreport:

```bash
PYTHONPATH=src python -m main analyze-search-space \
  --store memory \
  --benchmark-label v9a-curriculum \
  --task key_value_memory \
  --output-dir results
```

V9a-Phase-2-only Suchraumreport:

```bash
PYTHONPATH=src python -m main analyze-search-space \
  --store memory \
  --benchmark-label v9a-curriculum \
  --task key_value_memory \
  --curriculum-phase phase_2 \
  --output-dir results
```

V9a-Archivreport fuer `stateful_v2` im Retrieval-Raum:

```bash
PYTHONPATH=src python -m main analyze-archive \
  --store memory \
  --benchmark-label v9a-curriculum \
  --task key_value_memory \
  --variant stateful_v2 \
  --qd-profile retrieval_strategy \
  --output-dir results
```

V9a-Phase-2-only Archivreport fuer `stateful_v2`:

```bash
PYTHONPATH=src python -m main analyze-archive \
  --store memory \
  --benchmark-label v9a-curriculum \
  --task key_value_memory \
  --variant stateful_v2 \
  --qd-profile retrieval_strategy \
  --curriculum-phase phase_2 \
  --output-dir results
```

Sekundaerer Lauf mit historischem Sanity-Check:

```bash
PYTHONPATH=src python -m main benchmark-suite \
  --store memory \
  --tasks delayed_xor,bit_memory \
  --task-delay-sweep bit_memory=1,3,5,8 \
  --seeds 7,11,13,17,19 \
  --output-dir results
```

Der Lauf schreibt drei Dateien:

- `results/<label>.jsonl` mit einer Zeile pro Run
- `results/<label>.csv` mit der kompakten Aggregat-Tabelle
- `results/<label>.md` mit einer direkt zitierbaren Markdown-Tabelle

Die Aggregat-Summary enthaelt unter anderem:

- `success_rate`
- `mean_final_max_score`
- `mean_first_success_generation`
- `median_first_success_generation`
- `mean_best_node_count`
- `mean_best_enabled_connection_count`

### Online-Modus

```bash
PYTHONPATH=src python -m main run-online \
  --store memory \
  --config configs/base.yaml \
  --config configs/online.yaml \
  --config configs/event_memory.yaml \
  --config configs/local.yaml
```

### Online gegen SpacetimeDB

```bash
spacetime start
spacetime login --server-issued-login http://127.0.0.1:3000 --no-browser
cd spacetimedb
spacetime publish --server local neat-online-v4
cd ..
PYTHONPATH=src python -m main run-online \
  --store spacetimedb \
  --server-url http://127.0.0.1:3000 \
  --database-name neat-online-v4 \
  --config configs/base.yaml \
  --config configs/online.yaml \
  --config configs/event_memory.yaml \
  --config configs/local.yaml
```

### Online-Resume

```bash
PYTHONPATH=src python -m main run-online \
  --store spacetimedb \
  --server-url http://127.0.0.1:3000 \
  --database-name neat-online-v4 \
  --config configs/base.yaml \
  --config configs/online.yaml \
  --config configs/event_memory.yaml \
  --config configs/local.yaml \
  --resume-run-id <run_id>
```

### Online-Status

```bash
PYTHONPATH=src python -m main status-online <run_id> \
  --server-url http://127.0.0.1:3000 \
  --database-name neat-online-v4
```

Die Ausgabe zeigt unter anderem:

- aktive Population
- Rolling Best Score
- Rolling Avg Score
- Replacement Count
- Success Rate im Fenster
- Hall-of-Fame-Groesse

### Online-Benchmark

```bash
PYTHONPATH=src python -m main benchmark-online \
  --store memory \
  --config configs/base.yaml \
  --config configs/online.yaml \
  --config configs/event_memory.yaml \
  --config configs/local.yaml \
  --seeds 7,11,13
```

### Online-Compare

```bash
PYTHONPATH=src python -m main compare-online \
  --server-url http://127.0.0.1:3000 \
  --database-name neat-online-v4 \
  --task-name event_memory \
  --seed 7
```

### Hilfsskripte

```bash
bash ./scripts/run_online.sh
bash ./scripts/run_generation_benchmark.sh
bash ./scripts/run_benchmark.sh
bash ./scripts/simulate_crash_resume.sh <run_id> --store spacetimedb --server-url http://127.0.0.1:3000 --database-name neat-online-v4
```

## Tests

Schneller Kern-Check:

```bash
PYTHONPATH=src pytest tests/test_metrics.py tests/test_replacement.py tests/test_hall_of_fame.py tests/test_event_memory.py tests/test_online_repository.py tests/test_online_loop.py tests/test_resume_live_like.py
```

Live-SpacetimeDB-Checks:

```bash
PYTHONPATH=src pytest tests/test_spacetimedb_live.py
PYTHONPATH=src pytest tests/test_spacetimedb_online_live.py
```

## Wichtige V4-Dateien

- [online_loop.py](/mnt/c/Users/joach/NEAT.wsl.projekt/src/evolve/online_loop.py)
- [rtneat_scheduler.py](/mnt/c/Users/joach/NEAT.wsl.projekt/src/evolve/rtneat_scheduler.py)
- [replacement.py](/mnt/c/Users/joach/NEAT.wsl.projekt/src/evolve/replacement.py)
- [rolling_metrics.py](/mnt/c/Users/joach/NEAT.wsl.projekt/src/evolve/rolling_metrics.py)
- [event_memory.py](/mnt/c/Users/joach/NEAT.wsl.projekt/src/tasks/event_memory.py)
- [online_repository.py](/mnt/c/Users/joach/NEAT.wsl.projekt/src/db/online_repository.py)
- [queries.py](/mnt/c/Users/joach/NEAT.wsl.projekt/src/db/queries.py)
- [cli.py](/mnt/c/Users/joach/NEAT.wsl.projekt/src/ui/cli.py)
- [compare_report.py](/mnt/c/Users/joach/NEAT.wsl.projekt/src/ui/compare_report.py)
- [index.ts](/mnt/c/Users/joach/NEAT.wsl.projekt/spacetimedb/src/index.ts)

## Annahmen

- rtNEAT-artiges Verhalten wird in Python orchestriert; TensorNEAT selbst stellt hier keine native rtNEAT-API bereit.
- SpacetimeDB speichert koordinations- und konsistenzrelevanten Zustand, aber keine numerische Fitnessberechnung.
- Hall of Fame bleibt absichtlich Snapshot-basiert; archivierte Genome sind von aktiven Kandidaten strikt getrennt.
- Fuer sehr kleine lokale Benchmarks kann `JAX_PLATFORMS=cpu` unter WSL praktischer sein als GPU.

## Naechste sinnvolle Ausbauschritte

- Online-Benchmark direkt gegen SpacetimeDB ueber mehrere Seeds automatisieren
- Event-Memory in Richtung komplexerer Decision-Sequenzen erweitern
- echte Failure-Metriken wie `failed_jobs` und `duplicate_claim_prevented`
- kleine tabellarische CLI-Auswertung fuer Rolling Trends und Hall-of-Fame-Wachstum
