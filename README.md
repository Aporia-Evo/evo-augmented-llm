# Evo-Augmented LLM / Memory-Mechanics Projekt

## Zielsetzung

Dieses Projekt untersucht, wie sich **selektives Memory-Retrieval** in kleinen, neuroevolutionär optimierten Netzwerken realisieren lässt, **ohne** direkt auf volle Transformer-Attention oder große, gradientengetriebene Speicherarchitekturen zurückzugreifen.

Das Kernziel ist ein Mechanismus, der:

- **Information gezielt speichern** kann,
- **Distraktoren robust ignoriert**,
- **später den richtigen Value zum richtigen Key** wieder abruft,
- in einem **kleinen, statisch geformten JAX/NEAT-Setup** trainierbar bleibt,
- und **ohne expliziten Softmax-Addressing-Kollaps** auskommt.

Praktisch ist das Projekt eine schrittweise Suche nach der kleinsten evolvierbaren Architektur, die zwischen einfacher Rekurrenz und echter content-basierter Retrieval-Mechanik liegt.

---

## Leitfrage des Projekts

> Wie kommt man von einfachem rekurrentem Zustandsspeicher zu belastbarem Key-Value-Retrieval unter Distraktoren, ohne den Suchraum für Neuroevolution unbeherrschbar zu machen?

Daraus ergeben sich drei Daueranforderungen:

1. **mechanistische Interpretierbarkeit** statt Black-Box-Komplexität,
2. **minimal-invasive Architekturentwicklung** statt kompletter Redesigns,
3. **benchmark-getriebene Entscheidung** statt rein theoretischer Präferenz.

---

## Projektumgebung und Restriktionen

Das Projekt arbeitet unter für Memory-Mechaniken absichtlich harten Randbedingungen:

- kleine Populationen,
- neuroevolutionäre Optimierung statt Voll-Backpropagation,
- statische JAX-Shapes,
- reproduzierbare Benchmarks,
- geringer Architektur-Overhead,
- starke Sensitivität gegenüber Symmetrie, Default-Lösungen und lokalen Optima.

Diese Restriktionen sind nicht Nebenbedingungen, sondern Teil der Forschungsfrage: Gesucht wird **kein maximal mächtiges Modell**, sondern ein **unter Evolutionssuche stabil lernbarer Retrieval-Mechanismus**.

---

## Benchmark-Familien

Die Entwicklung wurde entlang mehrerer synthetischer Memory-Aufgaben validiert.

### 1. `bit_memory`
Einfachster Nachweis, dass eine Architektur überhaupt zeitverzögert Information halten kann.

**Zweck:**
- Basis-Check für rekurrente Zustandserhaltung
- Delay-Robustheit
- Debugging von Memory- und Evaluationspfaden

### 2. `key_value_memory_trivial`
Ein erster Übergang von reinem Delay-Memory zu expliziterem Zuordnungslernen.

**Zweck:**
- einfache Key-Value-Zuordnung
- erste Trennung von Store und Query
- Test, ob Architektur mehr als nur eine fortlaufende Zustandstrajektorie ausnutzt

### 3. `key_value_memory_easy` / `kv_easy`
Der zentrale Engpass-Benchmark des Projekts.

**Zweck:**
- mehrere Stores
- Distraktoren
- spätere Query
- exakter Abruf des richtigen Values

Diese Task ist die entscheidende Schwelle: Viele Mechaniken können Information grob halten, scheitern aber hier an **Key-Selektion, Distraktor-Unterdrückung oder stabilem Value-Readout**.

---

## Wie Fortschritt gemessen wird

Nicht nur der Gesamtscore ist relevant, sondern vor allem die mechanistischen Diagnosemetriken.

Wichtige Metriken im Projektverlauf:

- `query_key_match_score` – koppelt die Query überhaupt an den relevanten Key?
- `correct_key_selected` – wird der richtige Speicherinhalt adressiert?
- `correct_value_selected` – kommt tatsächlich der richtige Value heraus?
- `store_vs_distractor_beta_gap` – trennt der Schreibpfad relevante Stores von Distraktoren?
- `key_query_cosine_mean` / `key_query_cosine_at_query` – wie ähnlich sind Key- und Query-Räume?
- `key_variance_mean` / `query_variance_mean` – kollabieren Key/Query in zu geringe Varianz?
- `mean_memory_frobenius_norm` – bleibt der Speicher numerisch im stabilen Bereich?
- `query_memory_alignment` – ist überhaupt ein funktionaler content-basierter Read-Pfad vorhanden?

Diese Metriken sind zentral, weil viele Fehlschläge nicht in totalem Kollaps enden, sondern in **scheinbar aktiven, aber funktional entkoppelten Mechaniken**.

---

## Projektstationen bis jetzt

Die folgende Einordnung fasst die sichtbaren Projektstufen aus Code, Result-Dateien und bisherigen Experimenten zusammen. Sie ist auf die mechanistische Entwicklungslinie fokussiert.

### Phase A – frühe rekurrente Baselines (`v4.x` bis `v6`)

**Charakter:**
- einfache stateful Baselines
- rekurrente Speicherideen
- erste plastische oder zustandsbehaftete Varianten

**Ziel dieser Phase:**
- überhaupt verlässliche Zeitabhängigkeit herstellen
- Infrastruktur für Evaluation, Delay-Benchmarks und Reproduzierbarkeit festigen

**Ergebnis:**
- geeignet als Baseline für Delay-Memory
- nicht ausreichend für robustes Key-Value-Retrieval unter Distraktoren

**Wichtig:**
Diese Phase etablierte den Maßstab: Reine Rekurrenz reicht für kleine Memory-Tasks, aber nicht für selektives Retrieval.

---

### Phase B – Plastizität, Clamp- und Search-Space-Tuning (`v5a`, `v5b*`)

**Charakter:**
- Hebb-/AD-Plasticity-Experimente
- Clamp-Studien
- Eta- und Decay-Suchräume
- engere Suchraummodulation

**Ziel dieser Phase:**
- prüfen, ob lokale synaptische Anpassung Memory und Retrieval verbessern kann
- verstehen, wann Plastizität aktiv hilft und wann sie nur Instabilität oder Untersteuerung erzeugt

**Ergebnis:**
- wichtige Diagnosegewinne für Suchraumverhalten
- Plastizität häufig zu schwach, zu vorsichtig oder schlecht differenziert
- kein belastbarer Durchbruch für präzises Key-Value-Retrieval

**Lerneffekt:**
Plastizität alleine ersetzt keinen sauberen Retrieval-Mechanismus.

---

### Phase C – QD-Light, Archive- und Diversitätsarbeit (`v7`, `v7b`)

**Charakter:**
- QD-Light / Archive-Experimente
- stärkere Suchraumabdeckung
- Mechanismusdiversität statt nur Scoremaximierung

**Ziel dieser Phase:**
- nicht nur einen lokalen Elitepfad finden, sondern Suchraumregionen systematisch kartieren
- robuste Descriptoren für spätere Analysen etablieren

**Ergebnis:**
- bessere Sicht auf funktionale vs. degenerierte Lösungen
- hilfreich für Search-Space-Diagnostik
- aber noch kein echter Retrieval-Durchbruch

**Lerneffekt:**
Diversität im Suchraum hilft bei Analyse und Exploration, löst aber das mechanistische Kernproblem nicht selbst.

---

### Phase D – Curriculum- und Delay-Studien (`v8a`, `v8b`, `v8c`, `v8d`, `v9a`, `v9b`)

**Charakter:**
- Multi-Delay-Experimente
- Curriculum-Phasen
- Boundary-Studien
- KV-Easy-Smoke und Mid/Full-Smokes

**Ziel dieser Phase:**
- prüfen, ob schrittweise Aufgabenhärtung die Mechanik stabilisiert
- Delay-Generalisierung und Belastbarkeit testen

**Ergebnis:**
- Curricula helfen teilweise beim Trainingseinstieg
- die zentrale Retrieval-Barriere bleibt trotzdem bestehen
- besonders `key_value_memory_easy` exponiert die strukturellen Defizite sehr klar

**Lerneffekt:**
Curriculum kann Optimierung glätten, aber nicht die falsche Mechanik kompensieren.

---

### Phase E – KV-Diagnostik und Mechanismenzerlegung (`v11b`, `v11c`, implizit `v12`)

**Charakter:**
- stärkere Zerlegung in Write-/Read-/Match-Diagnostik
- explizitere Retrievalmetriken
- Fokus auf Key-/Value-Separation und Query-Match
- funktional starke Slot-Linie (`v12c_slots_readoutplus`) als Referenz

**Ziel dieser Phase:**
- herausfinden, **wo** genau frühere Varianten scheitern:
  - Write-Gate zu diffus?
  - Query-Match zu schwach?
  - Value-Readout instabil?
  - Store/Query-Kopplung verloren?

**Ergebnis:**
- sehr starker analytischer Fortschritt
- `v12c` war offenbar die funktional stärkste implizite Linie
- zeigte: verteilte, zustandsbasierte Mechanik ist evolvierbar, aber Retrievalpräzision bleibt begrenzt

**Lerneffekt:**
Das Projekt braucht einen Mechanismus zwischen reinem Stateful-Readout und hartem Addressing.

---

### Phase F – Explizites Addressing / adressierte Slots (`v13a`)

**Charakter:**
- Write-/Read-Addressing
- adressierte Slots
- stärkere Trennung von Controller und Speicherzugriff

**Ziel dieser Phase:**
- content-basiertes Retrieval explizit einführen
- die KV-Lücke mechanistisch direkter schließen

**Ergebnis:**
- klassischer Addressing-Kollaps
- diffuse oder symmetrische Adressierung
- lokale Default-Lösungen
- funktional schwach im Verhältnis zum Suchraumaufwand

**Lerneffekt:**
Explizites Softmax-artiges Routing ist für dieses Evolutionssetup zu fragil.

---

### Phase G – Delta-Memory-Linie (`stateful_v6_delta_memory`, `v14d`, `v14e`, `v14ff`, `v14g`)

**Charakter:**
- Übergang auf assoziative Fast-Weight-/Delta-Mechanik
- keine harte Speicherplatzwahl
- Retrieval via Matrix-Vektor-Operationen
- chirurgischeres Überschreiben durch Delta-Korrektur

**Ziel dieser Phase:**
- Store und Read content-basiert machen,
- aber Softmax-Addressing vermeiden,
- Distraktorresistenz erhöhen,
- einen kleinen, evolvierbaren Memory-Kern etablieren.

**Ergebnis bis vor V14h:**
- klar messbare `query_memory_alignment`-Signale
- numerisch stabile Speichergrößen
- Delta-Kern ist funktional plausibel
- aber Retrieval-Hauptmetriken bleiben begrenzt
- insbesondere schwach:
  - Query-Key-Match
  - klare Key-Selektion
  - saubere Trennung von Store vs. Distraktor im Beta-Gate

**Lerneffekt:**
Die Delta-Linie ist bislang die beste Brücke zwischen simpler Rekurrenz und Retrieval, aber noch nicht ausreichend entkoppelt im Key/Query-Raum.

---

### Phase H – V14h: bounded post-norm query decoupling

**Charakter:**
- minimal-invasive Änderung im bestehenden Delta-Pfad
- keine neue Architekturklasse
- kein Redesign des Memory-Kerns
- gezielte Query-Deflation relativ zur Key-Richtung

**V14h-Hypothese:**
Der verbleibende Engpass ist weiterhin ein **Symmetrie-/Parallelitätsproblem zwischen Key und Query**, nicht ein Mangel an Speicheraktualisierung.

**Konkrete Änderung:**
Nach Normalisierung von Key und Query wird die Query in begrenztem Ausmaß aus der Key-Richtung herausgedrückt:

- glatte Projektion auf zentrierten Vektoren,
- bounded via `tanh`,
- partielle Deflation statt harter Orthogonalisierung,
- danach erneute positive Normalisierung.

**Warum minimal-invasiv?**
- Delta-Update blieb unangetastet
- Beta-Gate-Logik blieb unangetastet
- keine neuen Slots
- kein neuer Variantentyp
- keine Änderung am grundlegenden Benchmark-Setup

**Ergebnis von V14h:**
- die neue Geometrie-/Decoupling-Telemetrie ist messbar,
- aber die Retrieval-Hauptmetriken zeigen noch **keinen klaren Durchbruch** gegenüber V14g/V14ff.

Beobachtete Kennzeichen:
- Decoupling-Effekt ist vorhanden,
- Key-/Query-Ähnlichkeit bleibt aber hoch,
- Key-/Query-Varianz ist klein,
- Speicher normiert stabil,
- Retrieval bleibt in der Hauptsache weiter begrenzt.

---

## Warum der bisher beste Pfad die Delta-Linie ist

Aus heutiger Sicht ist die Delta-Memory-Linie die aussichtsreichste Architekturklasse im Projekt, weil sie:

- **ohne explizites diskretes Addressing** auskommt,
- content-basiertes Retrieval **mathematisch direkt** ausdrückt,
- in JAX/Scan kompakt bleibt,
- mit den Restriktionen kleiner Neuroevolutions-Setups kompatibler ist als NTM/DNC/Slot-Softmax-Routing,
- und bereits funktionale Signale wie `query_memory_alignment` robust erzeugt.

Der offene Engpass ist nicht mehr „gibt es überhaupt einen Read/Write-Mechanismus?“, sondern:

> Wie bekommt man genügend **Asymmetrie, Varianz und selektive Gate-Differenzierung** in denselben kleinen evolvierbaren Kern?

---

## Was das Projekt ausdrücklich nicht will

Mehrfach sichtbar wurde, dass einige theoretisch attraktive Richtungen im gegebenen Setup schlechte Kandidaten sind.

### Kein voller Sprung zu O(T²)-Attention
Zu teuer, zu groß, zu weit weg von der eigentlichen Forschungsfrage.

### Kein NTM/DNC-artiges explizites Addressing
Zu fragile Softmax-/Routing-Dynamik für kleine evolutionäre Populationen.

### Kein reines additives Fast-Weight-Gedächtnis ohne Delta-Korrektur
Zu hohe Gefahr von Overload, Overschreiben und unlesbarem Assoziationsbrei.

### Kein großes Architektur-Redesign pro Iteration
Das Projekt lebt davon, Hypothesen **inkrementell** an derselben Grundlinie testbar zu machen.

---

## Aktueller Stand nach V14h

### Was bereits funktioniert
- reproduzierbare Benchmark-Suite
- solide Memory-/Retrieval-Diagnostik
- Archive-/Suchraumanalyse
- funktionaler Delta-Memory-Kern
- messbare inhaltsbasierte Auslese
- stabile Speicher-Normen ohne offensichtlichen numerischen Kollaps

### Was noch nicht gelöst ist
- klare Verbesserung bei `query_key_match_score`
- robuste Steigerung von `correct_key_selected`
- robuste Steigerung von `correct_value_selected`
- deutliche Differenz zwischen Store- und Distraktor-Beta
- ausreichende strukturelle Trennung von Key- und Query-Geometrie

### Ehrliche Kurzbewertung
Der Projektpfad ist **mechanistisch deutlich gereift**, aber das eigentliche Ziel – **belastbares selektives Retrieval unter Distraktoren** – ist noch nicht vollständig erreicht.

---

## Bisherige Hauptlehren

1. **Reine Rekurrenz reicht nicht.**
   Sie kann Delay-Memory, aber kein sauberes selektives KV-Retrieval unter Distraktoren.

2. **Softmax-Addressing ist im gegebenen Setup zu fragil.**
   Der V13-Pfad belegt das sehr klar.

3. **Delta-Memory ist der bisher beste mechanistische Kompromiss.**
   Es verbindet kleine Form, stabile JAX-Implementierung und inhaltliche Retrieval-Struktur.

4. **Der Engpass ist inzwischen geometrisch, nicht nur energetisch.**
   Speicheraktivität allein genügt nicht; Key- und Query-Räume müssen selektiv genug auseinandergezogen werden.

5. **Benchmark-getriebene Iteration funktioniert.**
   Auch wenn noch kein Durchbruch erreicht ist, ist heute wesentlich klarer, warum frühere Linien scheitern.

---

## Empfohlene Lesereihenfolge im Repo

Für einen schnellen Überblick über den bisherigen Stand:

1. `results/README.md`
2. ältere Basis- und Smoke-Reports (`v5*`, `v6`, `v7`, `v8*`, `v9*`)
3. KV-spezifische Reports (`v11*`, `v13a/*`)
4. Delta-Linie:
   - `results/v14e-delta.md`
   - `results/v14ff-delta.md`
   - `results/v14g-delta.md`
   - `results/v14h-delta.md`
5. Vergleichsberichte:
   - `results/v14e-vs-v14d-v14c-comparison.md`
   - `results/v14ff-vs-v14e-v14d-comparison.md`
   - `results/v14g-vs-v14ff-v14e-v14d-comparison.md`
   - `results/v14h-vs-v14g-v14ff-v14e-v14d-comparison.md`

---

## Kurzfazit

Das Projekt hat sich von einfachen rekurrenten Speicherexperimenten über Plastizitäts- und Curriculum-Studien, Slot- und Addressing-Ansätze hin zu einer **kompakten Delta-Memory-Linie** entwickelt. Diese Linie ist bislang die überzeugendste Kandidatin für das eigentliche Ziel:

> **kleines, evolvierbares, content-basiertes Memory-Retrieval ohne Softmax-Addressing-Kollaps**

V14h war ein sauberer, kleiner Schritt zur Entkopplung von Key und Query. Er verbessert die Diagnose und zeigt messbare Geometrieeffekte, aber noch keinen eindeutigen Retrieval-Durchbruch. Das Projekt steht damit aktuell an der Schwelle zwischen **funktionalem Delta-Readout** und **wirklich belastbarer selektiver Retrieval-Mechanik**.
