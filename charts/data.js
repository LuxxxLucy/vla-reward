// ── Shared color palette ──
const C = {
  b2: '#4e79a7', ens: '#e15759', b8: '#59a14f',
  vision: '#76b7b2', language: '#b07aa1',
  muted: '#6b7280', border: '#e0dfda', text: '#333'
};

// ── All experiment data ──
const DATA = {
  // Baseline: 7 videos, 3 methods
  baseline: [
    { video: 'Fold towel',       b2: 0.680, ens: 0.985, b8: 0.937 },
    { video: 'Pen → cup',        b2: 0.215, ens: 0.455, b8: 0.957 },
    { video: 'Remove cap',       b2: 0.771, ens: 0.806, b8: 0.863 },
    { video: 'Block → cup',      b2: 0.920, ens: 0.930, b8: 0.982 },
    { video: 'Marker → cup',     b2: 0.970, ens: 0.976, b8: 0.908 },
    { video: 'Stack cubes (LR)', b2: 0.884, ens: 0.894, b8: 0.799 },
    { video: 'Stack cubes (RS)', b2: 0.924, ens: 0.964, b8: 0.964 },
  ],

  // Reward curves — 4 highlighted videos, 10 frames
  curves: {
    fold_towel: {
      b2:  [-1.875, -1.875, -1.5, -0.875, -0.875, -0.625, -0.75, -0.625, -0.875, -0.875],
      ens: [-3.125, -3.042, -2.75, -2.333, -2.25, -2.25, -2.208, -2.042, -2.125, -2.0],
      b8:  [-3.125, -2.125, -1.125, -0.75, -0.375, -0.25, -0.25, -0.25, -0.25, -0.25],
      voc: { b2: 0.680, ens: 0.985, b8: 0.937 }
    },
    put_pen_cup: {
      b2:  [-2.25, -3.875, -4.75, -4.375, -4.125, -3.375, -4.25, -3.5, -3.375, -3.375],
      ens: [-3.333, -4.292, -4.75, -4.625, -4.542, -3.875, -4.083, -3.458, -3.292, -3.375],
      b8:  [-1.25, -2.0, -1.875, -1.0, -0.625, -0.5, -0.5, -0.375, -0.375, -0.25],
      voc: { b2: 0.215, ens: 0.455, b8: 0.957 }
    },
    remove_cap: {
      b2:  [-2.125, -1.875, -1.25, -1.125, -0.875, -0.875, -1.0, -1.125, -1.0, -0.75],
      ens: [-3.542, -3.125, -2.708, -2.417, -2.042, -2.0, -2.333, -2.125, -2.25, -1.75],
      b8:  [-4.625, -3.625, -2.0, -1.5, -1.875, -1.25, -1.375, -1.0, -1.125, -1.375],
      voc: { b2: 0.771, ens: 0.806, b8: 0.863 }
    },
    put_block_cup: {
      b2:  [-1.75, -1.25, -1.75, -1.375, -1.0, -1.0, -1.0, -0.875, -0.75, -0.875],
      ens: [-2.125, -1.958, -2.208, -1.958, -1.667, -1.542, -1.542, -1.458, -1.375, -1.458],
      b8:  [-3.75, -3.75, -2.25, -2.25, -1.375, -0.875, -0.5, -0.25, -0.25, -0.25],
      voc: { b2: 0.920, ens: 0.930, b8: 0.982 }
    }
  },

  // Reward-scope multi-method comparison — stackcubes, normalized progress
  landscape: {
    frames: [1,2,3,4,5,6,7,8,9,10],
    methods: [
      { name: 'TOPReward (8B)', color: '#59a14f',
        values: [0.0, 0.322, 0.282, 0.307, 0.275, 0.383, 0.405, 0.683, 0.867, 1.0] },
      { name: 'Our 2B', color: '#4e79a7',
        values: (function() {
          const r = [-1.75, -1.5, -1.0, -0.875, -1.125, -1.0, -0.625, -0.5, -0.375, -0.125];
          const mn = Math.min(...r), mx = Math.max(...r);
          return r.map(v => (v - mn) / (mx - mn));
        })() },
      { name: 'Our 2B ensemble', color: '#e15759',
        values: (function() {
          const r = [-2.75, -2.375, -1.958, -1.917, -2.083, -1.625, -1.375, -1.25, -1.167, -0.5];
          const mn = Math.min(...r), mx = Math.max(...r);
          return r.map(v => (v - mn) / (mx - mn));
        })() },
      { name: 'Robometer', color: '#f28e2b',
        values: [0.118, 0.410, 0.232, 0.429, 0.489, 0.708, 0.719, 0.829, 0.817, 0.862] },
      { name: 'BruteForce VLM', color: '#edc948',
        values: [0.0, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.6, 0.8, 0.8] },
      { name: 'GVL', color: '#bab0ac',
        values: [0.2, 0.4, 0.2, 0.4, 0.0, 0.6, 0.2, 0.8, 0.2, 1.0] },
    ]
  },

  // Timing breakdown — fold_towel (2B, 10 prefixes)
  // From results/timing_breakdown_fold_towel.json
  timing: {
    baseline_total: 60.3,
    cached_single_total: 58.2,
    cached_single_vision: 30.5,  // sum of per_prefix_vision
    cached_single_language: 27.7, // sum of per_prefix_language
    cached_ens3_total: 125.2,
    cached_ens3_vision: 32.1,   // sum of per_prefix_vision
    cached_ens3_language: 93.1,  // sum of per_prefix_language
    naive_3x: 181.0,
    cached_ens10_total: 640.1,
    cached_ens10_vision: 65.3,
    cached_ens10_language: 574.7,
    naive_10x: 582.0,
    b8_total: 683.4,
    b8_vision: 70.2,
    b8_language: 613.2,
    // Relative costs (baseline = 1×)
    mult_baseline: 1.0,
    mult_cached_ens: 2.07,
    mult_naive: 3.0,
    mult_8b: 11.8,
    // Vision fraction of cached single
    vision_frac: 0.52,
  },

  // Prompt heatmap: 10 prompts × 3 videos
  heatmap: {
    prompts: ['#0 Original', '#1 Attempting', '#2 Looking at', '#3 Performing',
              '#4 Intended task', '#5 Robot task', '#6 Progress on',
              '#7 Please confirm', '#8 The goal is', '#9 No failure'],
    videos: ['Pen → cup', 'Fold towel', 'Stack cubes'],
    values: [
      [0.215, 0.680, 0.924],
      [0.140, 0.935, 0.976],
      [0.923, 0.906, 0.795],
      [0.518, -0.175, 0.402],
      [-0.394, 0.895, 0.954],
      [0.900, 0.458, 0.960],
      [0.707, 0.038, 0.425],
      [0.927, 0.537, 0.444],
      [0.749, -0.809, 0.871],
      [0.952, 0.285, 0.862],
    ]
  }
};
