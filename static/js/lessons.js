/* Lessons page — side-by-side RL algorithm comparisons */
(function() {
    'use strict';

    // Color scheme matching model cards
    const ALGO_COLORS = {
        ppo: '#ff3ec9',
        dqn: '#39d9ff',
        a2c: '#f0b941',
    };

    // Known training results from HANDOFF.md
    const KNOWN_RESULTS = {
        basic:              { ppo: 80.2,  dqn: 71.1,  a2c: -300.8, ppo_noshape: 79.9 },
        defend_the_line:    { ppo: 17.3,  dqn: 16.5,  a2c: null,   ppo_noshape: 20.3 },
        defend_the_center:  { ppo: 9.6,   dqn: 4.8,   a2c: 0.8,    ppo_noshape: 12.6 },
        take_cover:         { ppo: 356.3, dqn: 337.7, a2c: 226.9,  ppo_noshape: 274.8 },
        health_gathering:   { ppo: 287.8, dqn: 413.3, a2c: null,   ppo_noshape: 315.7 },
        my_way_home:        { ppo: -2.6,  dqn: -2.6,  a2c: null,   ppo_noshape: 0.1 },
        deadly_corridor_s1: { ppo: 2042.4, dqn: 1732.1, a2c: null, ppo_noshape: 2070.9 },
        deadly_corridor_s3: { ppo: 1251.8, dqn: 935.0, a2c: null,  ppo_noshape: 1210.8 },
        deadly_corridor_s5: { ppo: 228.6, dqn: 185.9, a2c: null,   ppo_noshape: 223.0 },
    };

    let models = [];
    let algoChart = null;
    let shapingChart = null;

    // ─── Init ───
    document.addEventListener('DOMContentLoaded', function() {
        fetchModels();
        setupTabs();
        populateScenarioSelects();
        setupScenarioListeners();
    });

    function fetchModels() {
        fetch('/api/models')
            .then(r => r.json())
            .then(data => {
                models = data.models || [];
            })
            .catch(err => console.error('Failed to fetch models:', err));
    }

    // ─── Tab Navigation ───
    function setupTabs() {
        document.querySelectorAll('.lesson-tab').forEach(tab => {
            tab.addEventListener('click', function() {
                document.querySelectorAll('.lesson-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.lesson-panel').forEach(p => p.classList.remove('active'));
                this.classList.add('active');
                var lesson = this.getAttribute('data-lesson');
                document.getElementById('lesson-' + lesson).classList.add('active');
            });
        });
    }

    // ─── Scenario Dropdowns ───
    function populateScenarioSelects() {
        var scenarios = Object.keys(KNOWN_RESULTS);
        ['algoScenarioSelect', 'shapingScenarioSelect'].forEach(function(id) {
            var select = document.getElementById(id);
            if (!select) return;
            scenarios.forEach(function(s) {
                var opt = document.createElement('option');
                opt.value = s;
                opt.textContent = s.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
                select.appendChild(opt);
            });
        });
    }

    function setupScenarioListeners() {
        var algoSelect = document.getElementById('algoScenarioSelect');
        if (algoSelect) {
            algoSelect.addEventListener('change', function() {
                renderAlgoComparison(this.value);
            });
        }
        var shapingSelect = document.getElementById('shapingScenarioSelect');
        if (shapingSelect) {
            shapingSelect.addEventListener('change', function() {
                renderShapingComparison(this.value);
            });
        }
    }

    // ─── Algorithm Comparison ───
    function renderAlgoComparison(scenario) {
        var grid = document.getElementById('algoCompareGrid');
        if (!scenario || !KNOWN_RESULTS[scenario]) {
            grid.innerHTML = '<p style="color:var(--text-secondary)">Select a scenario to compare algorithms.</p>';
            return;
        }
        var data = KNOWN_RESULTS[scenario];
        var html = '';
        ['ppo', 'dqn', 'a2c'].forEach(function(algo) {
            var reward = data[algo];
            var available = reward !== null;
            html += '<div class="compare-card" style="border-color:' + ALGO_COLORS[algo] + '">';
            html += '<div class="compare-card-header" style="background:' + ALGO_COLORS[algo] + '20">';
            html += '<h3 style="color:' + ALGO_COLORS[algo] + '">' + algo.toUpperCase() + '</h3>';
            html += '</div>';
            html += '<div class="compare-card-body">';
            if (available) {
                html += '<div class="compare-reward">' + reward.toFixed(1) + '</div>';
                html += '<div class="compare-label">Mean Reward</div>';
            } else {
                html += '<div class="compare-reward dim">N/A</div>';
                html += '<div class="compare-label">Diverged / Not trained</div>';
            }
            html += '</div></div>';
        });
        grid.innerHTML = html;

        // Render chart
        renderAlgoChart(scenario, data);
    }

    function renderAlgoChart(scenario, data) {
        var ctx = document.getElementById('algoCompareChart');
        if (algoChart) algoChart.destroy();

        // Generate synthetic training curves based on final rewards
        var datasets = [];
        ['ppo', 'dqn', 'a2c'].forEach(function(algo) {
            if (data[algo] === null) return;
            var finalReward = data[algo];
            // Synthetic curve: starts near 0, converges to final reward
            var steps = [];
            var rewards = [];
            for (var i = 0; i <= 20; i++) {
                steps.push(i * 5000);
                // Logistic-ish growth with noise
                var progress = 1 - Math.exp(-0.2 * i);
                var noise = (Math.random() - 0.5) * Math.abs(finalReward) * 0.1;
                rewards.push(finalReward * progress + noise);
            }
            datasets.push({
                label: algo.toUpperCase(),
                data: rewards,
                borderColor: ALGO_COLORS[algo],
                backgroundColor: ALGO_COLORS[algo] + '20',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4,
            });
        });

        algoChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 21}, function(_, i) { return (i * 5000).toLocaleString(); }),
                datasets: datasets,
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: 'Training Curves — ' + scenario.replace(/_/g, ' '), color: '#e0e0e0' },
                    legend: { labels: { color: '#e0e0e0' } },
                    subtitle: { display: true, text: '(Synthetic curves from final rewards — connect W&B for real data)', color: '#888' },
                },
                scales: {
                    x: { title: { display: true, text: 'Timesteps', color: '#aaa' }, ticks: { color: '#888' }, grid: { color: '#333' } },
                    y: { title: { display: true, text: 'Mean Reward', color: '#aaa' }, ticks: { color: '#888' }, grid: { color: '#333' } },
                },
            },
        });
    }

    // ─── Reward Shaping Comparison ───
    function renderShapingComparison(scenario) {
        var grid = document.getElementById('shapingCompareGrid');
        if (!scenario || !KNOWN_RESULTS[scenario]) {
            grid.innerHTML = '<p style="color:var(--text-secondary)">Select a scenario to compare.</p>';
            return;
        }
        var data = KNOWN_RESULTS[scenario];
        var shaped = data.ppo;
        var unshaped = data.ppo_noshape;
        var delta = shaped - unshaped;
        var verdict = delta > 10 ? 'HELPS' : delta < -10 ? 'HURTS' : 'NEGLIGIBLE';
        var verdictColor = delta > 10 ? '#4ade80' : delta < -10 ? '#f87171' : '#fbbf24';

        var html = '';
        // Shaped card
        html += '<div class="compare-card" style="border-color:#ff3ec9">';
        html += '<div class="compare-card-header" style="background:#ff3ec920">';
        html += '<h3 style="color:#ff3ec9">PPO (Shaped)</h3></div>';
        html += '<div class="compare-card-body">';
        html += '<div class="compare-reward">' + shaped.toFixed(1) + '</div>';
        html += '<div class="compare-label">Mean Reward</div></div></div>';
        // Unshaped card
        html += '<div class="compare-card" style="border-color:#a78bfa">';
        html += '<div class="compare-card-header" style="background:#a78bfa20">';
        html += '<h3 style="color:#a78bfa">PPO (No Shaping)</h3></div>';
        html += '<div class="compare-card-body">';
        html += '<div class="compare-reward">' + unshaped.toFixed(1) + '</div>';
        html += '<div class="compare-label">Mean Reward</div></div></div>';
        // Delta card
        html += '<div class="compare-card" style="border-color:' + verdictColor + '">';
        html += '<div class="compare-card-header" style="background:' + verdictColor + '20">';
        html += '<h3 style="color:' + verdictColor + '">' + verdict + '</h3></div>';
        html += '<div class="compare-card-body">';
        html += '<div class="compare-reward" style="color:' + verdictColor + '">' + (delta >= 0 ? '+' : '') + delta.toFixed(1) + '</div>';
        html += '<div class="compare-label">Shaping Δ</div></div></div>';

        grid.innerHTML = html;

        // Render shaping chart
        renderShapingChart(scenario, shaped, unshaped);
    }

    function renderShapingChart(scenario, shaped, unshaped) {
        var ctx = document.getElementById('shapingCompareChart');
        if (shapingChart) shapingChart.destroy();

        function synthCurve(final) {
            var rewards = [];
            for (var i = 0; i <= 20; i++) {
                var progress = 1 - Math.exp(-0.2 * i);
                var noise = (Math.random() - 0.5) * Math.abs(final) * 0.08;
                rewards.push(final * progress + noise);
            }
            return rewards;
        }

        shapingChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 21}, function(_, i) { return (i * 5000).toLocaleString(); }),
                datasets: [
                    { label: 'PPO (Shaped)', data: synthCurve(shaped), borderColor: '#ff3ec9', borderWidth: 2, pointRadius: 0, tension: 0.4 },
                    { label: 'PPO (No Shaping)', data: synthCurve(unshaped), borderColor: '#a78bfa', borderWidth: 2, pointRadius: 0, tension: 0.4, borderDash: [5, 5] },
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: 'Shaped vs Unshaped — ' + scenario.replace(/_/g, ' '), color: '#e0e0e0' },
                    legend: { labels: { color: '#e0e0e0' } },
                    subtitle: { display: true, text: '(Synthetic curves — connect W&B for real data)', color: '#888' },
                },
                scales: {
                    x: { title: { display: true, text: 'Timesteps', color: '#aaa' }, ticks: { color: '#888' }, grid: { color: '#333' } },
                    y: { title: { display: true, text: 'Mean Reward', color: '#aaa' }, ticks: { color: '#888' }, grid: { color: '#333' } },
                },
            },
        });
    }
})();
