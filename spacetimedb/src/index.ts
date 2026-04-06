import { schema, table, t } from "spacetimedb/server";

const runs = table(
  { name: "runs", public: true },
  {
    run_id: t.string().primaryKey(),
    task_name: t.string(),
    mode: t.string(),
    seed: t.i32(),
    status: t.string(),
    config_json: t.string(),
    created_at: t.string(),
    finished_at: t.string(),
  },
);

const generations = table(
  { name: "generations", public: true },
  {
    generation_key: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    generation_id: t.i32(),
    state: t.string(),
    best_candidate_id: t.string(),
    best_score: t.f64(),
    avg_score: t.f64(),
    created_at: t.string(),
    committed_at: t.string(),
    eval_duration_ms: t.i32(),
    commit_duration_ms: t.i32(),
  },
);

const candidates = table(
  { name: "candidates", public: true },
  {
    candidate_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    generation_id: t.i32(),
    genome_blob: t.string(),
    status: t.string(),
    parent_ids_json: t.string(),
    created_at: t.string(),
  },
);

const fitness = table(
  { name: "fitness", public: true },
  {
    candidate_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    generation_id: t.i32(),
    score: t.f64(),
    raw_metrics_json: t.string(),
    evaluated_at: t.string(),
  },
);

const elite_archive = table(
  { name: "elite_archive", public: true },
  {
    elite_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    source_generation: t.i32(),
    candidate_id: t.string(),
    rank: t.i32(),
    score: t.f64(),
    frozen_genome_blob: t.string(),
    archived_at: t.string(),
  },
);

const events = table(
  { name: "events", public: true },
  {
    event_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    type: t.string(),
    payload_json: t.string(),
    created_at: t.string(),
  },
);

const checkpoints = table(
  { name: "checkpoints", public: true },
  {
    checkpoint_key: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    generation_id: t.i32(),
    state_blob: t.string(),
    parent_ids_json: t.string(),
    created_at: t.string(),
  },
);

const active_candidates = table(
  { name: "active_candidates", public: true },
  {
    candidate_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    slot_index: t.i32().index("btree"),
    variant: t.string(),
    genome_blob: t.string(),
    status: t.string(),
    rolling_score: t.f64(),
    eval_count: t.i32(),
    birth_step: t.i32(),
    last_eval_at: t.string(),
    parent_ids_json: t.string(),
    created_at: t.string(),
  },
);

const evaluation_jobs = table(
  { name: "evaluation_jobs", public: true },
  {
    job_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    candidate_id: t.string().index("btree"),
    task_payload_json: t.string(),
    status: t.string(),
    claimed_by: t.string(),
    created_at: t.string(),
    claimed_at: t.string(),
    finished_at: t.string(),
  },
);

const evaluation_results = table(
  { name: "evaluation_results", public: true },
  {
    result_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    candidate_id: t.string().index("btree"),
    score: t.f64(),
    raw_metrics_json: t.string(),
    created_at: t.string(),
  },
);

const hall_of_fame = table(
  { name: "hall_of_fame", public: true },
  {
    entry_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    candidate_id: t.string().index("btree"),
    score: t.f64(),
    frozen_genome_blob: t.string(),
    inserted_at: t.string(),
  },
);

const candidate_lifecycle_events = table(
  { name: "candidate_lifecycle_events", public: true },
  {
    event_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    candidate_id: t.string().index("btree"),
    event_type: t.string(),
    payload_json: t.string(),
    created_at: t.string(),
  },
);

const online_metrics = table(
  { name: "online_metrics", public: true },
  {
    metric_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    timestamp: t.string(),
    active_population_size: t.i32(),
    rolling_best_score: t.f64(),
    rolling_avg_score: t.f64(),
    replacement_count: t.i32(),
    success_rate_window: t.f64(),
  },
);

const online_state = table(
  { name: "online_state", public: true },
  {
    run_id: t.string().primaryKey(),
    step: t.i32(),
    replacement_count: t.i32(),
    success_window_json: t.string(),
    adapter_state_blob: t.string(),
    created_at: t.string(),
    updated_at: t.string(),
  },
);

const candidate_features = table(
  { name: "candidate_features", public: true },
  {
    candidate_id: t.string().primaryKey(),
    run_id: t.string().index("btree"),
    benchmark_label: t.string().index("btree"),
    task_name: t.string().index("btree"),
    delay_steps: t.i32().index("btree"),
    variant: t.string().index("btree"),
    seed: t.i32(),
    generation: t.i32(),
    hof_flag: t.bool().index("btree"),
    success: t.bool().index("btree"),
    final_max_score: t.f64(),
    first_success_generation: t.i32(),
    mean_alpha: t.f64(),
    std_alpha: t.f64(),
    mean_eta: t.f64(),
    std_eta: t.f64(),
    mean_plastic_d: t.f64(),
    std_plastic_d: t.f64(),
    plastic_d_at_lower_bound_fraction: t.f64(),
    plastic_d_at_zero_fraction: t.f64(),
    node_count: t.i32(),
    enabled_conn_count: t.i32(),
    mean_abs_delta_w: t.f64(),
    max_abs_delta_w: t.f64(),
    clamp_hit_rate: t.f64(),
    plasticity_active_fraction: t.f64(),
    mean_abs_fast_state: t.f64(),
    mean_abs_slow_state: t.f64(),
    slow_fast_contribution_ratio: t.f64(),
    mean_abs_decay_term: t.f64(),
    max_abs_decay_term: t.f64(),
    decay_effect_ratio: t.f64(),
    decay_near_zero_fraction: t.f64(),
    score_delay_3: t.f64(),
    score_delay_5: t.f64(),
    score_delay_8: t.f64(),
    success_delay_3: t.bool(),
    success_delay_5: t.bool(),
    success_delay_8: t.bool(),
    mean_score_over_delays: t.f64(),
    delay_score_std: t.f64(),
    delay_score_range: t.f64(),
    curriculum_enabled: t.bool(),
    curriculum_phase_1_delays: t.string(),
    curriculum_phase_2_delays: t.string(),
    curriculum_switch_generation: t.i32(),
    curriculum_phase: t.string(),
    active_evaluation_delays: t.string(),
    score_current_phase: t.f64(),
    query_accuracy: t.f64(),
    retrieval_score: t.f64(),
    exact_match_success: t.bool(),
    mean_query_distance: t.f64(),
    distractor_load: t.f64(),
    num_stores: t.f64(),
    num_queries: t.f64(),
    num_distractors: t.f64(),
    retrieval_margin: t.f64(),
    retrieval_confusion_rate: t.f64(),
    relevant_token_retention: t.f64(),
    query_response_margin: t.f64(),
    distractor_suppression_ratio: t.f64(),
    correct_key_selected: t.f64(),
    correct_value_selected: t.f64(),
    query_key_match_score: t.f64(),
    value_margin: t.f64(),
    distractor_competition_score: t.f64(),
    mean_abs_fast_state_during_store: t.f64(),
    mean_abs_slow_state_during_store: t.f64(),
    mean_abs_fast_state_during_query: t.f64(),
    mean_abs_slow_state_during_query: t.f64(),
    mean_abs_fast_state_during_distractor: t.f64(),
    mean_abs_slow_state_during_distractor: t.f64(),
    slow_query_coupling: t.f64(),
    store_query_state_gap: t.f64(),
    slow_fast_retrieval_ratio: t.f64(),
    retrieval_state_alignment: t.f64(),
  },
);

const candidate_feature_vectors = table(
  { name: "candidate_feature_vectors", public: true },
  {
    vector_key: t.string().primaryKey(),
    candidate_id: t.string().index("btree"),
    feature_version: t.string().index("btree"),
    vector_json: t.string(),
    norm_l2: t.f64(),
  },
);

const archive_cells = table(
  { name: "archive_cells", public: true },
  {
    archive_id: t.string().primaryKey(),
    benchmark_label: t.string().index("btree"),
    task_name: t.string().index("btree"),
    delay_steps: t.i32().index("btree"),
    variant: t.string().index("btree"),
    qd_profile: t.string().index("btree"),
    descriptor_schema_version: t.string(),
    descriptor_key: t.string().index("btree"),
    descriptor_values_json: t.string(),
    elite_candidate_id: t.string().index("btree"),
    elite_score: t.f64(),
    elite_run_id: t.string().index("btree"),
    updated_at: t.string(),
    curriculum_enabled: t.bool(),
    curriculum_phase_1_delays: t.string(),
    curriculum_phase_2_delays: t.string(),
    curriculum_switch_generation: t.i32(),
  },
);

const archive_events = table(
  { name: "archive_events", public: true },
  {
    event_id: t.string().primaryKey(),
    archive_id: t.string().index("btree"),
    benchmark_label: t.string().index("btree"),
    task_name: t.string().index("btree"),
    delay_steps: t.i32().index("btree"),
    variant: t.string().index("btree"),
    qd_profile: t.string().index("btree"),
    descriptor_schema_version: t.string(),
    descriptor_key: t.string().index("btree"),
    candidate_id: t.string().index("btree"),
    event_type: t.string().index("btree"),
    score: t.f64(),
    created_at: t.string(),
    curriculum_enabled: t.bool(),
    curriculum_phase_1_delays: t.string(),
    curriculum_phase_2_delays: t.string(),
    curriculum_switch_generation: t.i32(),
  },
);

const candidateInput = t.object("candidate_input", {
  candidate_id: t.string(),
  genome_blob: t.string(),
  status: t.string(),
  parent_ids_json: t.string(),
  created_at: t.string(),
});

const activeCandidateInput = t.object("active_candidate_input", {
  candidate_id: t.string(),
  slot_index: t.i32(),
  variant: t.string(),
  genome_blob: t.string(),
  status: t.string(),
  rolling_score: t.f64(),
  eval_count: t.i32(),
  birth_step: t.i32(),
  last_eval_at: t.string(),
  parent_ids_json: t.string(),
  created_at: t.string(),
});

const candidateFeatureInput = t.object("candidate_feature_input", {
  candidate_id: t.string(),
  run_id: t.string(),
  benchmark_label: t.string(),
  task_name: t.string(),
  delay_steps: t.i32(),
  variant: t.string(),
  seed: t.i32(),
  generation: t.i32(),
  hof_flag: t.bool(),
  success: t.bool(),
  final_max_score: t.f64(),
  first_success_generation: t.i32(),
  mean_alpha: t.f64(),
  std_alpha: t.f64(),
  mean_eta: t.f64(),
  std_eta: t.f64(),
  mean_plastic_d: t.f64(),
  std_plastic_d: t.f64(),
  plastic_d_at_lower_bound_fraction: t.f64(),
  plastic_d_at_zero_fraction: t.f64(),
  node_count: t.i32(),
  enabled_conn_count: t.i32(),
  mean_abs_delta_w: t.f64(),
  max_abs_delta_w: t.f64(),
  clamp_hit_rate: t.f64(),
  plasticity_active_fraction: t.f64(),
  mean_abs_fast_state: t.f64(),
  mean_abs_slow_state: t.f64(),
  slow_fast_contribution_ratio: t.f64(),
  mean_abs_decay_term: t.f64(),
  max_abs_decay_term: t.f64(),
  decay_effect_ratio: t.f64(),
  decay_near_zero_fraction: t.f64(),
  score_delay_3: t.f64(),
  score_delay_5: t.f64(),
  score_delay_8: t.f64(),
  success_delay_3: t.bool(),
  success_delay_5: t.bool(),
  success_delay_8: t.bool(),
  mean_score_over_delays: t.f64(),
  delay_score_std: t.f64(),
  delay_score_range: t.f64(),
  curriculum_enabled: t.bool(),
  curriculum_phase_1_delays: t.string(),
  curriculum_phase_2_delays: t.string(),
  curriculum_switch_generation: t.i32(),
  curriculum_phase: t.string(),
  active_evaluation_delays: t.string(),
  score_current_phase: t.f64(),
  query_accuracy: t.f64(),
  retrieval_score: t.f64(),
  exact_match_success: t.bool(),
  mean_query_distance: t.f64(),
  distractor_load: t.f64(),
  num_stores: t.f64(),
  num_queries: t.f64(),
  num_distractors: t.f64(),
  retrieval_margin: t.f64(),
  retrieval_confusion_rate: t.f64(),
  relevant_token_retention: t.f64(),
  query_response_margin: t.f64(),
  distractor_suppression_ratio: t.f64(),
  correct_key_selected: t.f64(),
  correct_value_selected: t.f64(),
  query_key_match_score: t.f64(),
  value_margin: t.f64(),
  distractor_competition_score: t.f64(),
  mean_abs_fast_state_during_store: t.f64(),
  mean_abs_slow_state_during_store: t.f64(),
  mean_abs_fast_state_during_query: t.f64(),
  mean_abs_slow_state_during_query: t.f64(),
  mean_abs_fast_state_during_distractor: t.f64(),
  mean_abs_slow_state_during_distractor: t.f64(),
  slow_query_coupling: t.f64(),
  store_query_state_gap: t.f64(),
  slow_fast_retrieval_ratio: t.f64(),
  retrieval_state_alignment: t.f64(),
});

const candidateFeatureVectorInput = t.object("candidate_feature_vector_input", {
  candidate_id: t.string(),
  feature_version: t.string(),
  vector_json: t.string(),
  norm_l2: t.f64(),
});

const archiveCellInput = t.object("archive_cell_input", {
  archive_id: t.string(),
  benchmark_label: t.string(),
  task_name: t.string(),
  delay_steps: t.i32(),
  variant: t.string(),
  qd_profile: t.string(),
  descriptor_schema_version: t.string(),
  descriptor_key: t.string(),
  descriptor_values_json: t.string(),
  elite_candidate_id: t.string(),
  elite_score: t.f64(),
  elite_run_id: t.string(),
  updated_at: t.string(),
  curriculum_enabled: t.bool(),
  curriculum_phase_1_delays: t.string(),
  curriculum_phase_2_delays: t.string(),
  curriculum_switch_generation: t.i32(),
});

const spacetimedb = schema(
  runs,
  generations,
  candidates,
  fitness,
  elite_archive,
  events,
  checkpoints,
  active_candidates,
  evaluation_jobs,
  evaluation_results,
  hall_of_fame,
  candidate_lifecycle_events,
  online_metrics,
  online_state,
  candidate_features,
  candidate_feature_vectors,
  archive_cells,
  archive_events,
);

function generationKey(runId: string, generationId: number): string {
  return `${runId}:g:${generationId}`;
}

function checkpointKey(runId: string, generationId: number): string {
  return `${runId}:checkpoint:${generationId}`;
}

function vectorKey(candidateId: string, featureVersion: string): string {
  return `${candidateId}:vector:${featureVersion}`;
}

function countRows(rows: Iterable<unknown>): number {
  let count = 0;
  for (const _row of rows) {
    count += 1;
  }
  return count;
}

function eventId(ctx: any, runId: string): string {
  return `${runId}:event:${countRows(ctx.db.events.iter()) + 1}`;
}

function candidateLifecycleEventId(ctx: any, runId: string): string {
  return `${runId}:candidate-event:${countRows(ctx.db.candidateLifecycleEvents.iter()) + 1}`;
}

function emitEvent(
  ctx: any,
  runId: string,
  type: string,
  payload: Record<string, unknown>,
  createdAt: string,
): void {
  ctx.db.events.insert({
    event_id: eventId(ctx, runId),
    run_id: runId,
    type,
    payload_json: JSON.stringify(payload),
    created_at: createdAt,
  });
}

function emitCandidateLifecycleEvent(
  ctx: any,
  runId: string,
  candidateId: string,
  eventType: string,
  payload: Record<string, unknown>,
  createdAt: string,
): void {
  ctx.db.candidateLifecycleEvents.insert({
    event_id: candidateLifecycleEventId(ctx, runId),
    run_id: runId,
    candidate_id: candidateId,
    event_type: eventType,
    payload_json: JSON.stringify(payload),
    created_at: createdAt,
  });
}

function getEliteTopK(configJson: string): number {
  try {
    const parsed = JSON.parse(configJson);
    const value = Number(parsed?.run?.elite_top_k ?? 3);
    if (!Number.isFinite(value) || value < 1) {
      return 3;
    }
    return Math.floor(value);
  } catch {
    return 3;
  }
}

function ensureGeneration(ctx: any, runId: string, generationId: number, createdAt: string): void {
  const key = generationKey(runId, generationId);
  const existing = ctx.db.generations.generation_key.find(key);
  if (!existing) {
    ctx.db.generations.insert({
      generation_key: key,
      run_id: runId,
      generation_id: generationId,
      state: "evaluating",
      best_candidate_id: "",
      best_score: Number.NaN,
      avg_score: Number.NaN,
      created_at: createdAt,
      committed_at: "",
      eval_duration_ms: -1,
      commit_duration_ms: -1,
    });
  }
}

function createRunRow(
  ctx: any,
  runId: string,
  taskName: string,
  seed: number,
  configJson: string,
  createdAt: string,
  mode: string,
): void {
  if (ctx.db.runs.run_id.find(runId)) {
    throw new Error(`Run already exists: ${runId}`);
  }
  ctx.db.runs.insert({
    run_id: runId,
    task_name: taskName,
    mode,
    seed,
    status: "running",
    config_json: configJson,
    created_at: createdAt,
    finished_at: "",
  });
  emitEvent(ctx, runId, "run_created", { task_name: taskName, seed, mode }, createdAt);
}

function requireRun(ctx: any, runId: string): any {
  const run = ctx.db.runs.run_id.find(runId);
  if (!run) {
    throw new Error(`Run not found: ${runId}`);
  }
  return run;
}

function requireActiveCandidate(ctx: any, candidateId: string): any {
  const candidate = ctx.db.activeCandidates.candidate_id.find(candidateId);
  if (!candidate) {
    throw new Error(`Active candidate not found: ${candidateId}`);
  }
  return candidate;
}

function cancelOpenJobsForCandidate(ctx: any, candidateId: string, finishedAt: string): void {
  for (const job of ctx.db.evaluationJobs.iter()) {
    if (job.candidate_id !== candidateId) {
      continue;
    }
    if (job.status === "finished" || job.status === "failed" || job.status === "cancelled") {
      continue;
    }
    job.status = "cancelled";
    job.finished_at = finishedAt;
    ctx.db.evaluationJobs.job_id.update(job);
  }
}

spacetimedb.reducer(
  "append_event",
  {
    run_id: t.string(),
    type: t.string(),
    payload_json: t.string(),
    created_at: t.string(),
  },
  (ctx, { run_id, type, payload_json, created_at }) => {
    let payload: Record<string, unknown> = {};
    try {
      payload = JSON.parse(payload_json) as Record<string, unknown>;
    } catch {
      payload = { raw_payload_json: payload_json };
    }
    emitEvent(ctx, run_id, type, payload, created_at);
  },
);

spacetimedb.reducer(
  "create_run",
  {
    run_id: t.string(),
    task_name: t.string(),
    seed: t.i32(),
    config_json: t.string(),
    created_at: t.string(),
    mode: t.string(),
  },
  (ctx, { run_id, task_name, seed, config_json, created_at, mode }) => {
    createRunRow(ctx, run_id, task_name, seed, config_json, created_at, mode);
  },
);

spacetimedb.reducer(
  "create_online_run",
  {
    run_id: t.string(),
    task_name: t.string(),
    seed: t.i32(),
    config_json: t.string(),
    created_at: t.string(),
  },
  (ctx, { run_id, task_name, seed, config_json, created_at }) => {
    createRunRow(ctx, run_id, task_name, seed, config_json, created_at, "online");
  },
);

spacetimedb.reducer(
  "insert_population",
  {
    run_id: t.string(),
    generation_id: t.i32(),
    candidates: t.array(candidateInput),
  },
  (ctx, { run_id, generation_id, candidates: incomingCandidates }) => {
    requireRun(ctx, run_id);
    const createdAt = incomingCandidates[0]?.created_at ?? "";
    ensureGeneration(ctx, run_id, generation_id, createdAt);
    for (const candidate of incomingCandidates) {
      ctx.db.candidates.insert({
        candidate_id: candidate.candidate_id,
        run_id,
        generation_id,
        genome_blob: candidate.genome_blob,
        status: candidate.status,
        parent_ids_json: candidate.parent_ids_json,
        created_at: candidate.created_at,
      });
    }
    emitEvent(
      ctx,
      run_id,
      "population_inserted",
      { generation_id, candidate_count: incomingCandidates.length },
      createdAt,
    );
  },
);

spacetimedb.reducer(
  "record_fitness",
  {
    candidate_id: t.string(),
    run_id: t.string(),
    generation_id: t.i32(),
    score: t.f64(),
    raw_metrics_json: t.string(),
    evaluated_at: t.string(),
  },
  (ctx, { candidate_id, run_id, generation_id, score, raw_metrics_json, evaluated_at }) => {
    const candidate = ctx.db.candidates.candidate_id.find(candidate_id);
    if (!candidate) {
      throw new Error(`Candidate not found: ${candidate_id}`);
    }
    const existing = ctx.db.fitness.candidate_id.find(candidate_id);
    if (existing) {
      ctx.db.fitness.candidate_id.update({
        candidate_id,
        run_id,
        generation_id,
        score,
        raw_metrics_json,
        evaluated_at,
      });
    } else {
      ctx.db.fitness.insert({
        candidate_id,
        run_id,
        generation_id,
        score,
        raw_metrics_json,
        evaluated_at,
      });
    }
    candidate.status = "evaluated";
    ctx.db.candidates.candidate_id.update(candidate);
  },
);

spacetimedb.reducer(
  "mark_generation_ready",
  {
    run_id: t.string(),
    generation_id: t.i32(),
    eval_duration_ms: t.i32(),
  },
  (ctx, { run_id, generation_id, eval_duration_ms }) => {
    const generation = ctx.db.generations.generation_key.find(generationKey(run_id, generation_id));
    if (!generation) {
      throw new Error(`Generation not found: ${generation_id}`);
    }
    generation.state = "ready";
    generation.eval_duration_ms = eval_duration_ms;
    ctx.db.generations.generation_key.update(generation);
    emitEvent(ctx, run_id, "generation_ready", { generation_id, eval_duration_ms }, generation.created_at);
  },
);

spacetimedb.reducer(
  "commit_generation",
  {
    run_id: t.string(),
    generation_id: t.i32(),
    committed_at: t.string(),
    commit_duration_ms: t.i32(),
  },
  (ctx, { run_id, generation_id, committed_at, commit_duration_ms }) => {
    const generation = ctx.db.generations.generation_key.find(generationKey(run_id, generation_id));
    const run = ctx.db.runs.run_id.find(run_id);
    if (!generation || !run) {
      throw new Error(`Missing run or generation for ${run_id} / ${generation_id}`);
    }
    if (generation.state === "committed") {
      if (commit_duration_ms >= 0) {
        generation.commit_duration_ms = commit_duration_ms;
        ctx.db.generations.generation_key.update(generation);
      }
      return;
    }
    const scored = [];
    for (const candidate of ctx.db.candidates.iter()) {
      if (candidate.run_id !== run_id || candidate.generation_id !== generation_id) {
        continue;
      }
      const fitnessRow = ctx.db.fitness.candidate_id.find(candidate.candidate_id);
      if (!fitnessRow) {
        throw new Error(`Missing fitness for candidate ${candidate.candidate_id}`);
      }
      scored.push({ candidate, fitness: fitnessRow });
    }
    if (scored.length === 0) {
      throw new Error(`No scored candidates for generation ${generation_id}`);
    }
    scored.sort((left, right) => {
      if (right.fitness.score !== left.fitness.score) {
        return right.fitness.score - left.fitness.score;
      }
      return left.candidate.candidate_id.localeCompare(right.candidate.candidate_id);
    });

    const best = scored[0];
    const avgScore = scored.reduce((sum, item) => sum + item.fitness.score, 0) / scored.length;
    const eliteTopK = getEliteTopK(run.config_json);
    const elites = scored.slice(0, eliteTopK);
    for (let index = 0; index < elites.length; index += 1) {
      const item = elites[index];
      ctx.db.eliteArchive.insert({
        elite_id: `${run_id}-g${generation_id.toString().padStart(4, "0")}-elite-${(index + 1)
          .toString()
          .padStart(2, "0")}`,
        run_id,
        source_generation: generation_id,
        candidate_id: item.candidate.candidate_id,
        rank: index + 1,
        score: item.fitness.score,
        frozen_genome_blob: `${item.candidate.genome_blob}`,
        archived_at: committed_at,
      });
    }

    generation.state = "committed";
    generation.best_candidate_id = best.candidate.candidate_id;
    generation.best_score = best.fitness.score;
    generation.avg_score = avgScore;
    generation.committed_at = committed_at;
    generation.commit_duration_ms = commit_duration_ms;
    ctx.db.generations.generation_key.update(generation);

    emitEvent(
      ctx,
      run_id,
      "generation_committed",
      {
        generation_id,
        best_candidate_id: best.candidate.candidate_id,
        best_score: best.fitness.score,
        avg_score: avgScore,
        elite_candidate_ids: elites.map((item) => item.candidate.candidate_id),
        eval_duration_ms: generation.eval_duration_ms,
        commit_duration_ms,
      },
      committed_at,
    );
  },
);

spacetimedb.reducer(
  "upsert_checkpoint",
  {
    run_id: t.string(),
    generation_id: t.i32(),
    state_blob: t.string(),
    parent_ids_json: t.string(),
    created_at: t.string(),
  },
  (ctx, { run_id, generation_id, state_blob, parent_ids_json, created_at }) => {
    const key = checkpointKey(run_id, generation_id);
    const existing = ctx.db.checkpoints.checkpoint_key.find(key);
    if (existing) {
      existing.state_blob = state_blob;
      existing.parent_ids_json = parent_ids_json;
      existing.created_at = created_at;
      ctx.db.checkpoints.checkpoint_key.update(existing);
    } else {
      ctx.db.checkpoints.insert({
        checkpoint_key: key,
        run_id,
        generation_id,
        state_blob,
        parent_ids_json,
        created_at,
      });
    }
    emitEvent(ctx, run_id, "checkpoint_saved", { generation_id }, created_at);
  },
);

spacetimedb.reducer(
  "create_next_generation",
  {
    run_id: t.string(),
    next_generation_id: t.i32(),
    offspring: t.array(candidateInput),
  },
  (ctx, { run_id, next_generation_id, offspring }) => {
    ensureGeneration(ctx, run_id, next_generation_id, offspring[0]?.created_at ?? "");
    for (const candidate of offspring) {
      ctx.db.candidates.insert({
        candidate_id: candidate.candidate_id,
        run_id,
        generation_id: next_generation_id,
        genome_blob: candidate.genome_blob,
        status: candidate.status,
        parent_ids_json: candidate.parent_ids_json,
        created_at: candidate.created_at,
      });
    }
    emitEvent(
      ctx,
      run_id,
      "next_generation_created",
      { generation_id: next_generation_id, candidate_count: offspring.length },
      offspring[0]?.created_at ?? "",
    );
  },
);

spacetimedb.reducer(
  "seed_active_population",
  {
    run_id: t.string(),
    candidates: t.array(activeCandidateInput),
  },
  (ctx, { run_id, candidates: incomingCandidates }) => {
    requireRun(ctx, run_id);
    for (const candidate of incomingCandidates) {
      ctx.db.activeCandidates.insert({
        candidate_id: candidate.candidate_id,
        run_id,
        slot_index: candidate.slot_index,
        variant: candidate.variant,
        genome_blob: candidate.genome_blob,
        status: candidate.status,
        rolling_score: candidate.rolling_score,
        eval_count: candidate.eval_count,
        birth_step: candidate.birth_step,
        last_eval_at: candidate.last_eval_at,
        parent_ids_json: candidate.parent_ids_json,
        created_at: candidate.created_at,
      });
      emitCandidateLifecycleEvent(
        ctx,
        run_id,
        candidate.candidate_id,
        "candidate_created",
        { slot_index: candidate.slot_index },
        candidate.created_at,
      );
    }
    emitEvent(
      ctx,
      run_id,
      "active_population_seeded",
      { candidate_count: incomingCandidates.length },
      incomingCandidates[0]?.created_at ?? "",
    );
  },
);

spacetimedb.reducer(
  "enqueue_evaluation",
  {
    job_id: t.string(),
    run_id: t.string(),
    candidate_id: t.string(),
    task_payload_json: t.string(),
    created_at: t.string(),
  },
  (ctx, { job_id, run_id, candidate_id, task_payload_json, created_at }) => {
    const candidate = requireActiveCandidate(ctx, candidate_id);
    ctx.db.evaluationJobs.insert({
      job_id,
      run_id,
      candidate_id,
      task_payload_json,
      status: "queued",
      claimed_by: "",
      created_at,
      claimed_at: "",
      finished_at: "",
    });
    candidate.status = "queued";
    ctx.db.activeCandidates.candidate_id.update(candidate);
    emitCandidateLifecycleEvent(ctx, run_id, candidate_id, "evaluation_enqueued", { job_id }, created_at);
  },
);

spacetimedb.reducer(
  "claim_job",
  {
    job_id: t.string(),
    worker_id: t.string(),
    claimed_at: t.string(),
  },
  (ctx, { job_id, worker_id, claimed_at }) => {
    const job = ctx.db.evaluationJobs.job_id.find(job_id);
    if (!job) {
      throw new Error(`Job not found: ${job_id}`);
    }
    if (job.status === "claimed" && job.claimed_by === worker_id) {
      return;
    }
    if (job.status !== "queued") {
      return;
    }
    job.status = "claimed";
    job.claimed_by = worker_id;
    job.claimed_at = claimed_at;
    ctx.db.evaluationJobs.job_id.update(job);
    const candidate = requireActiveCandidate(ctx, job.candidate_id);
    candidate.status = "evaluating";
    ctx.db.activeCandidates.candidate_id.update(candidate);
    emitCandidateLifecycleEvent(
      ctx,
      job.run_id,
      job.candidate_id,
      "evaluation_claimed",
      { job_id, worker_id },
      claimed_at,
    );
  },
);

spacetimedb.reducer(
  "submit_result",
  {
    result_id: t.string(),
    job_id: t.string(),
    candidate_id: t.string(),
    score: t.f64(),
    raw_metrics_json: t.string(),
    created_at: t.string(),
  },
  (ctx, { result_id, job_id, candidate_id, score, raw_metrics_json, created_at }) => {
    const job = ctx.db.evaluationJobs.job_id.find(job_id);
    if (!job) {
      throw new Error(`Job not found: ${job_id}`);
    }
    if (job.status !== "finished") {
      job.status = "finished";
      job.finished_at = created_at;
      ctx.db.evaluationJobs.job_id.update(job);
    }
    const existing = ctx.db.evaluationResults.result_id.find(result_id);
    if (existing) {
      existing.run_id = job.run_id;
      existing.candidate_id = candidate_id;
      existing.score = score;
      existing.raw_metrics_json = raw_metrics_json;
      existing.created_at = created_at;
      ctx.db.evaluationResults.result_id.update(existing);
    } else {
      ctx.db.evaluationResults.insert({
        result_id,
        run_id: job.run_id,
        candidate_id,
        score,
        raw_metrics_json,
        created_at,
      });
    }
    emitCandidateLifecycleEvent(
      ctx,
      job.run_id,
      candidate_id,
      "evaluation_finished",
      { job_id, score },
      created_at,
    );
  },
);

spacetimedb.reducer(
  "update_candidate_rolling_score",
  {
    candidate_id: t.string(),
    rolling_score: t.f64(),
    eval_count: t.i32(),
    last_eval_at: t.string(),
    status: t.string(),
  },
  (ctx, { candidate_id, rolling_score, eval_count, last_eval_at, status }) => {
    const candidate = requireActiveCandidate(ctx, candidate_id);
    candidate.rolling_score = rolling_score;
    candidate.eval_count = eval_count;
    candidate.last_eval_at = last_eval_at;
    candidate.status = status;
    ctx.db.activeCandidates.candidate_id.update(candidate);
    emitCandidateLifecycleEvent(
      ctx,
      candidate.run_id,
      candidate_id,
      "rolling_score_updated",
      { rolling_score, eval_count },
      last_eval_at,
    );
  },
);

spacetimedb.reducer(
  "spawn_offspring",
  {
    run_id: t.string(),
    candidate_id: t.string(),
    slot_index: t.i32(),
    variant: t.string(),
    genome_blob: t.string(),
    status: t.string(),
    rolling_score: t.f64(),
    eval_count: t.i32(),
    birth_step: t.i32(),
    last_eval_at: t.string(),
    parent_ids_json: t.string(),
    created_at: t.string(),
  },
  (
    ctx,
    {
      run_id,
      candidate_id,
      slot_index,
      variant,
      genome_blob,
      status,
      rolling_score,
      eval_count,
      birth_step,
      last_eval_at,
      parent_ids_json,
      created_at,
    },
  ) => {
    ctx.db.activeCandidates.insert({
      candidate_id,
      run_id,
      slot_index,
      variant,
      genome_blob,
      status,
      rolling_score,
      eval_count,
      birth_step,
      last_eval_at,
      parent_ids_json,
      created_at,
    });
    emitCandidateLifecycleEvent(
      ctx,
      run_id,
      candidate_id,
      "candidate_spawned",
      { slot_index, parent_ids: JSON.parse(parent_ids_json) },
      created_at,
    );
  },
);

spacetimedb.reducer(
  "retire_candidate",
  {
    candidate_id: t.string(),
    retired_at: t.string(),
  },
  (ctx, { candidate_id, retired_at }) => {
    const candidate = requireActiveCandidate(ctx, candidate_id);
    candidate.status = "retired";
    ctx.db.activeCandidates.candidate_id.update(candidate);
    cancelOpenJobsForCandidate(ctx, candidate_id, retired_at);
    emitCandidateLifecycleEvent(
      ctx,
      candidate.run_id,
      candidate_id,
      "candidate_retired",
      { slot_index: candidate.slot_index },
      retired_at,
    );
  },
);

spacetimedb.reducer(
  "activate_candidate",
  {
    candidate_id: t.string(),
    activated_at: t.string(),
  },
  (ctx, { candidate_id, activated_at }) => {
    const candidate = requireActiveCandidate(ctx, candidate_id);
    candidate.status = "active";
    ctx.db.activeCandidates.candidate_id.update(candidate);
    emitCandidateLifecycleEvent(
      ctx,
      candidate.run_id,
      candidate_id,
      "candidate_activated",
      { slot_index: candidate.slot_index },
      activated_at,
    );
  },
);

spacetimedb.reducer(
  "promote_to_hall_of_fame",
  {
    entry_id: t.string(),
    run_id: t.string(),
    candidate_id: t.string(),
    score: t.f64(),
    frozen_genome_blob: t.string(),
    inserted_at: t.string(),
  },
  (ctx, { entry_id, run_id, candidate_id, score, frozen_genome_blob, inserted_at }) => {
    for (const entry of ctx.db.hallOfFame.iter()) {
      if (entry.run_id === run_id && entry.candidate_id === candidate_id) {
        return;
      }
    }
    ctx.db.hallOfFame.insert({
      entry_id,
      run_id,
      candidate_id,
      score,
      frozen_genome_blob,
      inserted_at,
    });
    emitCandidateLifecycleEvent(
      ctx,
      run_id,
      candidate_id,
      "candidate_promoted_to_hall_of_fame",
      { entry_id, score },
      inserted_at,
    );
  },
);

spacetimedb.reducer(
  "upsert_candidate_features",
  {
    feature: candidateFeatureInput,
  },
  (ctx, { feature }) => {
    const existing = ctx.db.candidateFeatures.candidate_id.find(feature.candidate_id);
    if (existing) {
      existing.run_id = feature.run_id;
      existing.benchmark_label = feature.benchmark_label;
      existing.task_name = feature.task_name;
      existing.delay_steps = feature.delay_steps;
      existing.variant = feature.variant;
      existing.seed = feature.seed;
      existing.generation = feature.generation;
      existing.hof_flag = feature.hof_flag;
      existing.success = feature.success;
      existing.final_max_score = feature.final_max_score;
      existing.first_success_generation = feature.first_success_generation;
      existing.mean_alpha = feature.mean_alpha;
      existing.std_alpha = feature.std_alpha;
      existing.mean_eta = feature.mean_eta;
      existing.std_eta = feature.std_eta;
      existing.mean_plastic_d = feature.mean_plastic_d;
      existing.std_plastic_d = feature.std_plastic_d;
      existing.plastic_d_at_lower_bound_fraction = feature.plastic_d_at_lower_bound_fraction;
      existing.plastic_d_at_zero_fraction = feature.plastic_d_at_zero_fraction;
      existing.node_count = feature.node_count;
      existing.enabled_conn_count = feature.enabled_conn_count;
      existing.mean_abs_delta_w = feature.mean_abs_delta_w;
      existing.max_abs_delta_w = feature.max_abs_delta_w;
      existing.clamp_hit_rate = feature.clamp_hit_rate;
      existing.plasticity_active_fraction = feature.plasticity_active_fraction;
      existing.mean_abs_fast_state = feature.mean_abs_fast_state;
      existing.mean_abs_slow_state = feature.mean_abs_slow_state;
      existing.slow_fast_contribution_ratio = feature.slow_fast_contribution_ratio;
      existing.mean_abs_decay_term = feature.mean_abs_decay_term;
      existing.max_abs_decay_term = feature.max_abs_decay_term;
      existing.decay_effect_ratio = feature.decay_effect_ratio;
      existing.decay_near_zero_fraction = feature.decay_near_zero_fraction;
      existing.score_delay_3 = feature.score_delay_3;
      existing.score_delay_5 = feature.score_delay_5;
      existing.score_delay_8 = feature.score_delay_8;
      existing.success_delay_3 = feature.success_delay_3;
      existing.success_delay_5 = feature.success_delay_5;
      existing.success_delay_8 = feature.success_delay_8;
      existing.mean_score_over_delays = feature.mean_score_over_delays;
      existing.delay_score_std = feature.delay_score_std;
      existing.delay_score_range = feature.delay_score_range;
      existing.curriculum_enabled = feature.curriculum_enabled;
      existing.curriculum_phase_1_delays = feature.curriculum_phase_1_delays;
      existing.curriculum_phase_2_delays = feature.curriculum_phase_2_delays;
      existing.curriculum_switch_generation = feature.curriculum_switch_generation;
      existing.curriculum_phase = feature.curriculum_phase;
      existing.active_evaluation_delays = feature.active_evaluation_delays;
      existing.score_current_phase = feature.score_current_phase;
      existing.query_accuracy = feature.query_accuracy;
      existing.retrieval_score = feature.retrieval_score;
      existing.exact_match_success = feature.exact_match_success;
      existing.mean_query_distance = feature.mean_query_distance;
      existing.distractor_load = feature.distractor_load;
      existing.num_stores = feature.num_stores;
      existing.num_queries = feature.num_queries;
      existing.num_distractors = feature.num_distractors;
      existing.retrieval_margin = feature.retrieval_margin;
      existing.retrieval_confusion_rate = feature.retrieval_confusion_rate;
      existing.relevant_token_retention = feature.relevant_token_retention;
      existing.query_response_margin = feature.query_response_margin;
      existing.distractor_suppression_ratio = feature.distractor_suppression_ratio;
      existing.correct_key_selected = feature.correct_key_selected;
      existing.correct_value_selected = feature.correct_value_selected;
      existing.query_key_match_score = feature.query_key_match_score;
      existing.value_margin = feature.value_margin;
      existing.distractor_competition_score = feature.distractor_competition_score;
      existing.mean_abs_fast_state_during_store = feature.mean_abs_fast_state_during_store;
      existing.mean_abs_slow_state_during_store = feature.mean_abs_slow_state_during_store;
      existing.mean_abs_fast_state_during_query = feature.mean_abs_fast_state_during_query;
      existing.mean_abs_slow_state_during_query = feature.mean_abs_slow_state_during_query;
      existing.mean_abs_fast_state_during_distractor = feature.mean_abs_fast_state_during_distractor;
      existing.mean_abs_slow_state_during_distractor = feature.mean_abs_slow_state_during_distractor;
      existing.slow_query_coupling = feature.slow_query_coupling;
      existing.store_query_state_gap = feature.store_query_state_gap;
      existing.slow_fast_retrieval_ratio = feature.slow_fast_retrieval_ratio;
      existing.retrieval_state_alignment = feature.retrieval_state_alignment;
      ctx.db.candidateFeatures.candidate_id.update(existing);
      return;
    }
    ctx.db.candidateFeatures.insert({
      candidate_id: feature.candidate_id,
      run_id: feature.run_id,
      benchmark_label: feature.benchmark_label,
      task_name: feature.task_name,
      delay_steps: feature.delay_steps,
      variant: feature.variant,
      seed: feature.seed,
      generation: feature.generation,
      hof_flag: feature.hof_flag,
      success: feature.success,
      final_max_score: feature.final_max_score,
      first_success_generation: feature.first_success_generation,
      mean_alpha: feature.mean_alpha,
      std_alpha: feature.std_alpha,
      mean_eta: feature.mean_eta,
      std_eta: feature.std_eta,
      mean_plastic_d: feature.mean_plastic_d,
      std_plastic_d: feature.std_plastic_d,
      plastic_d_at_lower_bound_fraction: feature.plastic_d_at_lower_bound_fraction,
      plastic_d_at_zero_fraction: feature.plastic_d_at_zero_fraction,
      node_count: feature.node_count,
      enabled_conn_count: feature.enabled_conn_count,
      mean_abs_delta_w: feature.mean_abs_delta_w,
      max_abs_delta_w: feature.max_abs_delta_w,
      clamp_hit_rate: feature.clamp_hit_rate,
      plasticity_active_fraction: feature.plasticity_active_fraction,
      mean_abs_fast_state: feature.mean_abs_fast_state,
      mean_abs_slow_state: feature.mean_abs_slow_state,
      slow_fast_contribution_ratio: feature.slow_fast_contribution_ratio,
      mean_abs_decay_term: feature.mean_abs_decay_term,
      max_abs_decay_term: feature.max_abs_decay_term,
      decay_effect_ratio: feature.decay_effect_ratio,
      decay_near_zero_fraction: feature.decay_near_zero_fraction,
      score_delay_3: feature.score_delay_3,
      score_delay_5: feature.score_delay_5,
      score_delay_8: feature.score_delay_8,
      success_delay_3: feature.success_delay_3,
      success_delay_5: feature.success_delay_5,
      success_delay_8: feature.success_delay_8,
      mean_score_over_delays: feature.mean_score_over_delays,
      delay_score_std: feature.delay_score_std,
      delay_score_range: feature.delay_score_range,
      curriculum_enabled: feature.curriculum_enabled,
      curriculum_phase_1_delays: feature.curriculum_phase_1_delays,
      curriculum_phase_2_delays: feature.curriculum_phase_2_delays,
      curriculum_switch_generation: feature.curriculum_switch_generation,
      curriculum_phase: feature.curriculum_phase,
      active_evaluation_delays: feature.active_evaluation_delays,
      score_current_phase: feature.score_current_phase,
      query_accuracy: feature.query_accuracy,
      retrieval_score: feature.retrieval_score,
      exact_match_success: feature.exact_match_success,
      mean_query_distance: feature.mean_query_distance,
      distractor_load: feature.distractor_load,
      num_stores: feature.num_stores,
      num_queries: feature.num_queries,
      num_distractors: feature.num_distractors,
      retrieval_margin: feature.retrieval_margin,
      retrieval_confusion_rate: feature.retrieval_confusion_rate,
      relevant_token_retention: feature.relevant_token_retention,
      query_response_margin: feature.query_response_margin,
      distractor_suppression_ratio: feature.distractor_suppression_ratio,
      correct_key_selected: feature.correct_key_selected,
      correct_value_selected: feature.correct_value_selected,
      query_key_match_score: feature.query_key_match_score,
      value_margin: feature.value_margin,
      distractor_competition_score: feature.distractor_competition_score,
      mean_abs_fast_state_during_store: feature.mean_abs_fast_state_during_store,
      mean_abs_slow_state_during_store: feature.mean_abs_slow_state_during_store,
      mean_abs_fast_state_during_query: feature.mean_abs_fast_state_during_query,
      mean_abs_slow_state_during_query: feature.mean_abs_slow_state_during_query,
      mean_abs_fast_state_during_distractor: feature.mean_abs_fast_state_during_distractor,
      mean_abs_slow_state_during_distractor: feature.mean_abs_slow_state_during_distractor,
      slow_query_coupling: feature.slow_query_coupling,
      store_query_state_gap: feature.store_query_state_gap,
      slow_fast_retrieval_ratio: feature.slow_fast_retrieval_ratio,
      retrieval_state_alignment: feature.retrieval_state_alignment,
    });
  },
);

spacetimedb.reducer(
  "mark_candidate_feature_hof",
  {
    candidate_id: t.string(),
    hof_flag: t.bool(),
  },
  (ctx, { candidate_id, hof_flag }) => {
    const feature = ctx.db.candidateFeatures.candidate_id.find(candidate_id);
    if (!feature) {
      return;
    }
    feature.hof_flag = hof_flag;
    ctx.db.candidateFeatures.candidate_id.update(feature);
  },
);

spacetimedb.reducer(
  "upsert_candidate_feature_vector",
  {
    vector: candidateFeatureVectorInput,
  },
  (ctx, { vector }) => {
    const key = vectorKey(vector.candidate_id, vector.feature_version);
    const existing = ctx.db.candidateFeatureVectors.vector_key.find(key);
    if (existing) {
      existing.candidate_id = vector.candidate_id;
      existing.feature_version = vector.feature_version;
      existing.vector_json = vector.vector_json;
      existing.norm_l2 = vector.norm_l2;
      ctx.db.candidateFeatureVectors.vector_key.update(existing);
      return;
    }
    ctx.db.candidateFeatureVectors.insert({
      vector_key: key,
      candidate_id: vector.candidate_id,
      feature_version: vector.feature_version,
      vector_json: vector.vector_json,
      norm_l2: vector.norm_l2,
    });
  },
);

spacetimedb.reducer(
  "consider_archive_candidate",
  {
    cell: archiveCellInput,
    event_id: t.string(),
  },
  (ctx, { cell, event_id }) => {
    const existing = ctx.db.archiveCells.archive_id.find(cell.archive_id);
    let eventType = "skip";
    if (!existing) {
      ctx.db.archiveCells.insert({
        archive_id: cell.archive_id,
        benchmark_label: cell.benchmark_label,
        task_name: cell.task_name,
        delay_steps: cell.delay_steps,
        variant: cell.variant,
        qd_profile: cell.qd_profile,
        descriptor_schema_version: cell.descriptor_schema_version,
        descriptor_key: cell.descriptor_key,
        descriptor_values_json: cell.descriptor_values_json,
        elite_candidate_id: cell.elite_candidate_id,
        elite_score: cell.elite_score,
        elite_run_id: cell.elite_run_id,
        updated_at: cell.updated_at,
        curriculum_enabled: cell.curriculum_enabled,
        curriculum_phase_1_delays: cell.curriculum_phase_1_delays,
        curriculum_phase_2_delays: cell.curriculum_phase_2_delays,
        curriculum_switch_generation: cell.curriculum_switch_generation,
      });
      eventType = "insert";
    } else if (cell.elite_score > existing.elite_score) {
      existing.benchmark_label = cell.benchmark_label;
      existing.task_name = cell.task_name;
      existing.delay_steps = cell.delay_steps;
      existing.variant = cell.variant;
      existing.qd_profile = cell.qd_profile;
      existing.descriptor_schema_version = cell.descriptor_schema_version;
      existing.descriptor_key = cell.descriptor_key;
      existing.descriptor_values_json = cell.descriptor_values_json;
      existing.elite_candidate_id = cell.elite_candidate_id;
      existing.elite_score = cell.elite_score;
      existing.elite_run_id = cell.elite_run_id;
      existing.updated_at = cell.updated_at;
      existing.curriculum_enabled = cell.curriculum_enabled;
      existing.curriculum_phase_1_delays = cell.curriculum_phase_1_delays;
      existing.curriculum_phase_2_delays = cell.curriculum_phase_2_delays;
      existing.curriculum_switch_generation = cell.curriculum_switch_generation;
      ctx.db.archiveCells.archive_id.update(existing);
      eventType = "replace";
    }
    ctx.db.archiveEvents.insert({
      event_id,
      archive_id: cell.archive_id,
      benchmark_label: cell.benchmark_label,
      task_name: cell.task_name,
      delay_steps: cell.delay_steps,
      variant: cell.variant,
      qd_profile: cell.qd_profile,
      descriptor_schema_version: cell.descriptor_schema_version,
      descriptor_key: cell.descriptor_key,
      candidate_id: cell.elite_candidate_id,
      event_type: eventType,
      score: cell.elite_score,
      created_at: cell.updated_at,
      curriculum_enabled: cell.curriculum_enabled,
      curriculum_phase_1_delays: cell.curriculum_phase_1_delays,
      curriculum_phase_2_delays: cell.curriculum_phase_2_delays,
      curriculum_switch_generation: cell.curriculum_switch_generation,
    });
  },
);

spacetimedb.reducer(
  "capture_online_metrics",
  {
    metric_id: t.string(),
    run_id: t.string(),
    timestamp: t.string(),
    active_population_size: t.i32(),
    rolling_best_score: t.f64(),
    rolling_avg_score: t.f64(),
    replacement_count: t.i32(),
    success_rate_window: t.f64(),
  },
  (
    ctx,
    {
      metric_id,
      run_id,
      timestamp,
      active_population_size,
      rolling_best_score,
      rolling_avg_score,
      replacement_count,
      success_rate_window,
    },
  ) => {
    ctx.db.onlineMetrics.insert({
      metric_id,
      run_id,
      timestamp,
      active_population_size,
      rolling_best_score,
      rolling_avg_score,
      replacement_count,
      success_rate_window,
    });
  },
);

spacetimedb.reducer(
  "upsert_online_state",
  {
    run_id: t.string(),
    step: t.i32(),
    replacement_count: t.i32(),
    success_window_json: t.string(),
    adapter_state_blob: t.string(),
    updated_at: t.string(),
  },
  (ctx, { run_id, step, replacement_count, success_window_json, adapter_state_blob, updated_at }) => {
    const existing = ctx.db.onlineState.run_id.find(run_id);
    if (existing) {
      existing.step = step;
      existing.replacement_count = replacement_count;
      existing.success_window_json = success_window_json;
      existing.adapter_state_blob = adapter_state_blob;
      existing.updated_at = updated_at;
      ctx.db.onlineState.run_id.update(existing);
      return;
    }
    ctx.db.onlineState.insert({
      run_id,
      step,
      replacement_count,
      success_window_json,
      adapter_state_blob,
      created_at: updated_at,
      updated_at,
    });
  },
);

spacetimedb.reducer(
  "resume_online_run",
  {
    run_id: t.string(),
    resumed_at: t.string(),
  },
  (ctx, { run_id, resumed_at }) => {
    const run = requireRun(ctx, run_id);
    if (run.status === "finished") {
      throw new Error(`Run already finished: ${run_id}`);
    }
    let requeuedJobCount = 0;
    let resumedCandidateCount = 0;
    for (const job of ctx.db.evaluationJobs.iter()) {
      if (job.run_id !== run_id || job.status !== "claimed") {
        continue;
      }
      job.status = "queued";
      job.claimed_by = "";
      job.claimed_at = "";
      ctx.db.evaluationJobs.job_id.update(job);
      requeuedJobCount += 1;
    }
    for (const candidate of ctx.db.activeCandidates.iter()) {
      if (candidate.run_id !== run_id || candidate.status !== "evaluating") {
        continue;
      }
      candidate.status = "active";
      ctx.db.activeCandidates.candidate_id.update(candidate);
      resumedCandidateCount += 1;
    }
    run.status = "running";
    ctx.db.runs.run_id.update(run);
    const state = ctx.db.onlineState.run_id.find(run_id);
    emitEvent(
      ctx,
      run_id,
      "run_resumed",
      {
        mode: "online",
        step: state?.step ?? 0,
        requeued_job_count: requeuedJobCount,
        resumed_candidate_count: resumedCandidateCount,
      },
      resumed_at,
    );
  },
);

spacetimedb.reducer(
  "finish_run",
  {
    run_id: t.string(),
    finished_at: t.string(),
  },
  (ctx, { run_id, finished_at }) => {
    const run = requireRun(ctx, run_id);
    run.status = "finished";
    run.finished_at = finished_at;
    ctx.db.runs.run_id.update(run);
    emitEvent(ctx, run_id, "run_finished", {}, finished_at);
  },
);

spacetimedb.reducer(
  "finish_online_run",
  {
    run_id: t.string(),
    finished_at: t.string(),
  },
  (ctx, { run_id, finished_at }) => {
    const run = requireRun(ctx, run_id);
    if (run.mode !== "online") {
      throw new Error(`Run is not online: ${run_id}`);
    }
    run.status = "finished";
    run.finished_at = finished_at;
    ctx.db.runs.run_id.update(run);
    emitEvent(ctx, run_id, "run_finished", { mode: "online" }, finished_at);
  },
);

export default spacetimedb;
