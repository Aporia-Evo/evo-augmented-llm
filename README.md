# Evo-Augmented LLM / Memory Mechanics Project

## Objective

This project investigates how **selective memory retrieval** can be implemented in small, neuroevolution-optimized networks **without** relying on full transformer attention or large gradient-driven memory architectures.

The core goal is a mechanism that can:

- **store information selectively**,
- **robustly ignore distractors**,
- **retrieve the correct value for the correct key later**,
- remain trainable in a **small, statically shaped JAX/NEAT setup**,
- and do so **without explicit softmax-addressing collapse**.

In practice, the project is a step-by-step search for the smallest evolvable architecture that lies between simple recurrence and genuine content-based retrieval mechanics.

---

## Guiding Project Question

> How do we move from simple recurrent state memory to reliable key-value retrieval under distractors, without making the neuroevolution search space unmanageable?

This leads to three persistent requirements:

1. **mechanistic interpretability** instead of black-box complexity,
2. **minimally invasive architectural development** instead of full redesigns,
3. **benchmark-driven decisions** instead of purely theoretical preferences.

---

## Project Environment and Constraints

The project operates under deliberately strict conditions for memory mechanisms:

- small populations,
- neuroevolutionary optimization instead of full backpropagation,
- static JAX shapes,
- reproducible benchmarks,
- low architecture overhead,
- high sensitivity to symmetry, default solutions, and local optima.

These constraints are not side conditions—they are part of the research question. The aim is **not a maximally powerful model**, but a **retrieval mechanism that learns stably under evolutionary search**.

---

## Benchmark Families

Development has been validated across several synthetic memory tasks.

### 1. `bit_memory`
The simplest proof that an architecture can retain information over time delays.

**Purpose:**
- baseline check for recurrent state retention
- delay robustness
- debugging memory and evaluation paths

### 2. `key_value_memory_trivial`
A first transition from pure delay memory to more explicit assignment learning.

**Purpose:**
- simple key-value assignment
- initial separation of store and query
- test whether the architecture does more than exploit a continuous state trajectory

### 3. `key_value_memory_easy` / `kv_easy`
The central bottleneck benchmark of the project.

**Purpose:**
- multiple stores
- distractors
- delayed query
- exact retrieval of the correct value

This task is the critical threshold: many mechanisms can roughly preserve information, but fail here due to **key selection, distractor suppression, or stable value readout**.

---

## How Progress Is Measured

Not only the total score matters, but especially the mechanistic diagnostic metrics.

Important metrics throughout the project:

- `query_key_match_score` – does the query couple to the relevant key at all?
- `correct_key_selected` – is the correct memory content addressed?
- `correct_value_selected` – is the correct value actually produced?
- `store_vs_distractor_beta_gap` – does the write path separate relevant stores from distractors?
- `key_query_cosine_mean` / `key_query_cosine_at_query` – how similar are key and query spaces?
- `key_variance_mean` / `query_variance_mean` – do key/query representations collapse to low variance?
- `mean_memory_frobenius_norm` – does memory remain numerically stable?
- `query_memory_alignment` – is there a functional content-based read path at all?

These metrics are central because many failures do not end in total collapse, but in **apparently active yet functionally decoupled mechanisms**.

---

## Project Stages So Far

The following classification summarizes visible project stages based on code, result files, and experiments so far. It focuses on the mechanistic development trajectory.

### Phase A – early recurrent baselines (`v4.x` to `v6`)

**Characteristics:**
- simple stateful baselines
- recurrent memory ideas
- first plastic or stateful variants

**Goal of this phase:**
- establish reliable temporal dependence at all
- strengthen infrastructure for evaluation, delay benchmarks, and reproducibility

**Outcome:**
- suitable as a baseline for delay memory
- not sufficient for robust key-value retrieval under distractors

**Key takeaway:**
This phase established the benchmark: pure recurrence is enough for small memory tasks, but not for selective retrieval.

---

### Phase B – plasticity, clamp, and search-space tuning (`v5a`, `v5b*`)

**Characteristics:**
- Hebbian / AD plasticity experiments
- clamp studies
- eta and decay search spaces
- tighter search-space modulation

**Goal of this phase:**
- test whether local synaptic adaptation can improve memory and retrieval
- understand when plasticity helps and when it only adds instability or under-control

**Outcome:**
- important diagnostic gains for search-space behavior
- plasticity often too weak, too cautious, or poorly differentiated
- no reliable breakthrough for precise key-value retrieval

**Key takeaway:**
Plasticity alone does not replace a clean retrieval mechanism.

---

### Phase C – QD-light, archive, and diversity work (`v7`, `v7b`)

**Characteristics:**
- QD-light / archive experiments
- broader search-space coverage
- mechanism diversity instead of pure score maximization

**Goal of this phase:**
- not only find one local elite path, but systematically map regions of the search space
- establish robust descriptors for later analyses

**Outcome:**
- better visibility into functional vs. degenerate solutions
- useful for search-space diagnostics
- still no true retrieval breakthrough

**Key takeaway:**
Search-space diversity helps analysis and exploration, but does not solve the mechanistic core problem by itself.

---

### Phase D – curriculum and delay studies (`v8a`, `v8b`, `v8c`, `v8d`, `v9a`, `v9b`)

**Characteristics:**
- multi-delay experiments
- curriculum phases
- boundary studies
- KV-easy smoke and mid/full smoke evaluations

**Goal of this phase:**
- test whether gradual task hardening stabilizes the mechanism
- test delay generalization and robustness

**Outcome:**
- curricula help partially with training entry
- the central retrieval barrier still remains
- `key_value_memory_easy` in particular exposes structural deficits very clearly

**Key takeaway:**
Curriculum can smooth optimization, but cannot compensate for the wrong mechanism.

---

### Phase E – KV diagnostics and mechanism decomposition (`v11b`, `v11c`, implicitly `v12`)

**Characteristics:**
- stronger decomposition into write/read/match diagnostics
- more explicit retrieval metrics
- focus on key/value separation and query match
- functionally strong slot line (`v12c_slots_readoutplus`) as reference

**Goal of this phase:**
- identify exactly **where** earlier variants fail:
  - write gate too diffuse?
  - query match too weak?
  - value readout unstable?
  - store/query coupling lost?

**Outcome:**
- very strong analytical progress
- `v12c` appears to have been the functionally strongest implicit line
- showed that distributed, state-based mechanics are evolvable, but retrieval precision stays limited

**Key takeaway:**
The project needs a mechanism between pure stateful readout and hard addressing.

---

### Phase F – explicit addressing / addressed slots (`v13a`)

**Characteristics:**
- write/read addressing
- addressed slots
- stronger separation of controller and memory access

**Goal of this phase:**
- introduce content-based retrieval explicitly
- close the KV gap more directly at the mechanism level

**Outcome:**
- classic addressing collapse
- diffuse or symmetric addressing
- local default solutions
- functionally weak relative to search-space cost

**Key takeaway:**
Explicit softmax-like routing is too fragile for this evolutionary setup.

---

### Phase G – delta-memory line (`stateful_v6_delta_memory`, `v14d`, `v14e`, `v14ff`, `v14g`)

**Characteristics:**
- shift to associative fast-weight / delta mechanics
- no hard memory-slot selection
- retrieval via matrix-vector operations
- more surgical overwrite via delta correction

**Goal of this phase:**
- make store and read content-based,
- avoid softmax addressing,
- increase distractor resistance,
- establish a small, evolvable memory core.

**Outcome up to pre-V14h:**
- clearly measurable `query_memory_alignment` signals
- numerically stable memory magnitudes
- delta core is functionally plausible
- but key retrieval metrics remain limited
- especially weak:
  - query-key match
  - clear key selection
  - clean store-vs-distractor separation in the beta gate

**Key takeaway:**
The delta line is currently the best bridge between simple recurrence and retrieval, but key/query space is still insufficiently decoupled.

---

### Phase H – V14h: bounded post-norm query decoupling

**Characteristics:**
- minimally invasive change in the existing delta path
- no new architecture class
- no redesign of the memory core
- targeted query deflation relative to key direction

**V14h hypothesis:**
The remaining bottleneck is still a **symmetry/parallelism issue between key and query**, not a lack of memory update capability.

**Concrete change:**
After normalization of key and query, query is pushed away from key direction within a bounded range:

- smooth projection on centered vectors,
- bounded via `tanh`,
- partial deflation instead of hard orthogonalization,
- followed by renewed positive normalization.

**Why minimally invasive?**
- delta update unchanged
- beta-gate logic unchanged
- no new slots
- no new variant type
- no change to the underlying benchmark setup

**Outcome of V14h:**
- new geometry/decoupling telemetry is measurable,
- but core retrieval metrics still show **no clear breakthrough** versus V14g/V14ff.

Observed indicators:
- decoupling effect is present,
- key/query similarity remains high,
- key/query variance is low,
- memory norm remains stable,
- retrieval remains mostly constrained.

---

## Why the Delta Line Is Currently the Best Path

From today’s perspective, the delta-memory line is the most promising architecture class in the project because it:

- works **without explicit discrete addressing**,
- expresses content-based retrieval **directly in mathematical form**,
- stays compact in JAX/Scan,
- is more compatible with small neuroevolution setups than NTM/DNC/slot-softmax routing,
- and already produces functional signals such as `query_memory_alignment` robustly.

The open bottleneck is no longer “is there any read/write mechanism at all?”, but:

> How do we get enough **asymmetry, variance, and selective gate differentiation** into the same small evolvable core?

---

## What the Project Explicitly Does Not Want

It became clear multiple times that some theoretically attractive directions are poor candidates in this setup.

### No full jump to O(T²) attention
Too expensive, too large, too far from the actual research question.

### No NTM/DNC-like explicit addressing
Softmax/routing dynamics are too fragile for small evolutionary populations.

### No pure additive fast-weight memory without delta correction
Too high a risk of overload, overwrite, and unreadable associative clutter.

### No large architecture redesign every iteration
The project depends on testing hypotheses **incrementally** on the same base line.

---

## Current Status After V14h

### What already works
- reproducible benchmark suite
- solid memory/retrieval diagnostics
- archive/search-space analysis
- functional delta-memory core
- measurable content-based readout
- stable memory norms without obvious numerical collapse

### What is not solved yet
- clear improvement in `query_key_match_score`
- robust increase in `correct_key_selected`
- robust increase in `correct_value_selected`
- strong separation between store and distractor beta
- sufficient structural separation of key and query geometry

### Honest short assessment
The project path has become **much more mature mechanistically**, but the core goal—**reliable selective retrieval under distractors**—has not been fully achieved yet.

---

## Main Lessons So Far

1. **Pure recurrence is not enough.**
   It can handle delay memory, but not clean selective KV retrieval under distractors.

2. **Softmax addressing is too fragile in this setup.**
   The V13 path demonstrates this very clearly.

3. **Delta memory is currently the best mechanistic compromise.**
   It combines compact form, stable JAX implementation, and meaningful retrieval structure.

4. **The bottleneck is now geometric, not just energetic.**
   Memory activity alone is insufficient; key and query spaces must be separated selectively enough.

5. **Benchmark-driven iteration works.**
   Even without a breakthrough yet, it is now much clearer why earlier lines failed.

---

## Recommended Reading Order in the Repo

For a quick overview of the current state:

1. `results/README.md`
2. older baseline and smoke reports (`v5*`, `v6`, `v7`, `v8*`, `v9*`)
3. KV-specific reports (`v11*`, `v13a/*`)
4. Delta line:
   - `results/v14e-delta.md`
   - `results/v14ff-delta.md`
   - `results/v14g-delta.md`
   - `results/v14h-delta.md`
5. Comparison reports:
   - `results/v14e-vs-v14d-v14c-comparison.md`
   - `results/v14ff-vs-v14e-v14d-comparison.md`
   - `results/v14g-vs-v14ff-v14e-v14d-comparison.md`
   - `results/v14h-vs-v14g-v14ff-v14e-v14d-comparison.md`

---

## Short Conclusion

The project evolved from simple recurrent memory experiments through plasticity and curriculum studies, slot and addressing approaches, and finally to a **compact delta-memory line**. So far, this line is the most convincing candidate for the core target:

> **small, evolvable, content-based memory retrieval without softmax-addressing collapse**

V14h was a clean, small step toward key/query decoupling. It improves diagnostics and shows measurable geometric effects, but not yet a clear retrieval breakthrough. The project currently stands at the threshold between **functional delta readout** and **truly robust selective retrieval mechanics**.
