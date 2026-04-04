/* ============================================================
   DOOM RL – Training Page JS
   ============================================================ */

(function () {
    'use strict';

    var socket = window.socket;

    // ---- DOM References ----
    var scenarioSelect = document.getElementById('scenarioSelect');
    var btnStart       = document.getElementById('btnStart');
    var btnPause       = document.getElementById('btnPause');
    var btnResume      = document.getElementById('btnResume');
    var btnStop        = document.getElementById('btnStop');
    var btnLoadDefaults= document.getElementById('btnLoadDefaults');
    var btnClearLog    = document.getElementById('btnClearLog');
    var statusBadge    = document.getElementById('trainingStatusBadge');
    var progressBar    = document.getElementById('trainingProgress');
    var timestepDisplay= document.getElementById('timestepDisplay');
    var fpsDisplay     = document.getElementById('fpsDisplay');
    var rewardDisplay  = document.getElementById('rewardDisplay');
    var epLenDisplay   = document.getElementById('epLenDisplay');
    var entropyDisplay = document.getElementById('entropyDisplay');
    var trainingLog    = document.getElementById('trainingLog');

    // Form inputs
    var inputs = {
        learningRate:   document.getElementById('learningRate'),
        nSteps:         document.getElementById('nSteps'),
        clipRange:      document.getElementById('clipRange'),
        gamma:          document.getElementById('gamma'),
        gaeLambda:      document.getElementById('gaeLambda'),
        totalTimesteps: document.getElementById('totalTimesteps'),
        killReward:     document.getElementById('killReward'),
        missPenalty:    document.getElementById('missPenalty'),
        stepPenalty:    document.getElementById('stepPenalty'),
        damagePenalty:  document.getElementById('damagePenalty'),
        ammoPenalty:    document.getElementById('ammoPenalty')
    };

    // Range display values
    var clipRangeVal = document.getElementById('clipRangeVal');
    var gammaVal     = document.getElementById('gammaVal');
    var gaeLambdaVal = document.getElementById('gaeLambdaVal');

    // Charts
    var rewardChart = null;
    var lossChart   = null;
    var scenarioDefaults = {};
    var isTraining = false;
    var updateCounter = 0;

    // ---- Range Slider Display Updates ----
    if (inputs.clipRange) {
        inputs.clipRange.addEventListener('input', function () {
            clipRangeVal.textContent = parseFloat(this.value).toFixed(2);
        });
    }
    if (inputs.gamma) {
        inputs.gamma.addEventListener('input', function () {
            gammaVal.textContent = parseFloat(this.value).toFixed(3);
        });
    }
    if (inputs.gaeLambda) {
        inputs.gaeLambda.addEventListener('input', function () {
            gaeLambdaVal.textContent = parseFloat(this.value).toFixed(2);
        });
    }

    // ---- Initialize Charts ----
    if (typeof createRewardChart === 'function') {
        rewardChart = createRewardChart('rewardChart');
        lossChart   = createLossChart('lossChart');
    }

    // ---- Fetch Scenarios ----
    fetch('/api/scenarios')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            scenarioSelect.innerHTML = '<option value="">-- Select Scenario --</option>';
            if (data.scenarios && data.scenarios.length > 0) {
                data.scenarios.forEach(function (s) {
                    var opt = document.createElement('option');
                    opt.value = s.id || s.name;
                    opt.textContent = s.name || s.id;
                    scenarioSelect.appendChild(opt);

                    // Store defaults
                    if (s.defaults) {
                        scenarioDefaults[s.id || s.name] = s.defaults;
                    }
                });
            }
        })
        .catch(function () {
            scenarioSelect.innerHTML = '<option value="">Failed to load scenarios</option>';
        });

    // ---- Load Defaults on Scenario Change ----
    scenarioSelect.addEventListener('change', function () {
        loadDefaultsForScenario(this.value);
    });

    btnLoadDefaults.addEventListener('click', function () {
        loadDefaultsForScenario(scenarioSelect.value);
        window.showToast('Defaults loaded for scenario.', 'info');
    });

    function loadDefaultsForScenario(scenarioId) {
        var d = scenarioDefaults[scenarioId];
        if (!d) return;

        if (d.learning_rate !== undefined) inputs.learningRate.value = d.learning_rate;
        if (d.n_steps !== undefined)       inputs.nSteps.value = d.n_steps;
        if (d.clip_range !== undefined) {
            inputs.clipRange.value = d.clip_range;
            clipRangeVal.textContent = parseFloat(d.clip_range).toFixed(2);
        }
        if (d.gamma !== undefined) {
            inputs.gamma.value = d.gamma;
            gammaVal.textContent = parseFloat(d.gamma).toFixed(3);
        }
        if (d.gae_lambda !== undefined) {
            inputs.gaeLambda.value = d.gae_lambda;
            gaeLambdaVal.textContent = parseFloat(d.gae_lambda).toFixed(2);
        }
        if (d.total_timesteps !== undefined) inputs.totalTimesteps.value = d.total_timesteps;
        if (d.kill_reward !== undefined)     inputs.killReward.value = d.kill_reward;
        if (d.miss_penalty !== undefined)    inputs.missPenalty.value = d.miss_penalty;
        if (d.step_penalty !== undefined)    inputs.stepPenalty.value = d.step_penalty;
        if (d.damage_penalty !== undefined)  inputs.damagePenalty.value = d.damage_penalty;
        if (d.ammo_penalty !== undefined)    inputs.ammoPenalty.value = d.ammo_penalty;
    }

    // ---- Start Training ----
    btnStart.addEventListener('click', function () {
        var scenario = scenarioSelect.value;
        if (!scenario) {
            window.showToast('Please select a scenario first.', 'warning');
            return;
        }

        var config = {
            scenario:        scenario,
            learning_rate:   parseFloat(inputs.learningRate.value),
            n_steps:         parseInt(inputs.nSteps.value),
            clip_range:      parseFloat(inputs.clipRange.value),
            gamma:           parseFloat(inputs.gamma.value),
            gae_lambda:      parseFloat(inputs.gaeLambda.value),
            total_timesteps: parseInt(inputs.totalTimesteps.value),
            rewards: {
                kill_reward:    parseFloat(inputs.killReward.value),
                miss_penalty:   parseFloat(inputs.missPenalty.value),
                step_penalty:   parseFloat(inputs.stepPenalty.value),
                damage_penalty: parseFloat(inputs.damagePenalty.value),
                ammo_penalty:   parseFloat(inputs.ammoPenalty.value)
            }
        };

        // Reset charts
        if (rewardChart) {
            rewardChart.data.labels = [];
            rewardChart.data.datasets[0].data = [];
            rewardChart.update('none');
        }
        if (lossChart) {
            lossChart.data.labels = [];
            lossChart.data.datasets[0].data = [];
            lossChart.update('none');
        }

        updateCounter = 0;
        logMessage('Starting training on scenario: ' + scenario, 'info');
        socket.emit('start_training', config);
    });

    // ---- Pause / Resume / Stop ----
    btnPause.addEventListener('click', function () {
        socket.emit('pause_training');
        logMessage('Pause requested...', 'info');
    });

    btnResume.addEventListener('click', function () {
        socket.emit('resume_training');
        logMessage('Resume requested...', 'info');
    });

    btnStop.addEventListener('click', function () {
        if (confirm('Stop current training? Progress will be saved as a checkpoint.')) {
            socket.emit('stop_training');
            logMessage('Stop requested...', 'warning');
        }
    });

    // ---- Training Status Events ----
    socket.on('training_status', function (data) {
        var status = data.status;
        statusBadge.textContent = status.charAt(0).toUpperCase() + status.slice(1);

        if (status === 'training' || status === 'running') {
            isTraining = true;
            statusBadge.className = 'mono text-success training-active';
            btnStart.disabled = true;
            btnPause.disabled = false;
            btnResume.style.display = 'none';
            btnPause.style.display = '';
            btnStop.disabled = false;
        } else if (status === 'paused') {
            statusBadge.className = 'mono text-warning';
            btnPause.style.display = 'none';
            btnResume.style.display = '';
            btnResume.disabled = false;
        } else if (status === 'stopped' || status === 'completed' || status === 'error') {
            isTraining = false;
            statusBadge.className = 'mono text-muted';
            btnStart.disabled = false;
            btnPause.disabled = true;
            btnResume.style.display = 'none';
            btnPause.style.display = '';
            btnStop.disabled = true;
            if (status === 'completed') {
                logMessage('Training completed!', 'success');
                window.showToast('Training completed successfully!', 'success');
            } else if (status === 'error') {
                logMessage('Training error: ' + (data.message || 'Unknown'), 'error');
                window.showToast('Training error: ' + (data.message || ''), 'error');
            }
        }
    });

    // ---- Training Metrics Events ----
    socket.on('training_metrics', function (data) {
        updateCounter++;

        // Progress
        var current = data.timesteps || data.total_timesteps_done || 0;
        var total   = data.total_timesteps || parseInt(inputs.totalTimesteps.value) || 1;
        var pct     = Math.min((current / total) * 100, 100);
        progressBar.style.width = pct.toFixed(1) + '%';
        if (!progressBar.classList.contains('active')) progressBar.classList.add('active');
        timestepDisplay.textContent = window.formatNumber(current, 0) + ' / ' + window.formatNumber(total, 0);

        // Stats
        if (data.fps !== undefined) fpsDisplay.textContent = Math.round(data.fps);
        if (data.ep_reward_mean !== undefined) rewardDisplay.textContent = window.formatNumber(data.ep_reward_mean);
        if (data.ep_len_mean !== undefined) epLenDisplay.textContent = window.formatNumber(data.ep_len_mean, 0);
        if (data.entropy_loss !== undefined) entropyDisplay.textContent = window.formatNumber(data.entropy_loss, 4);

        // Update charts (every data point)
        var label = window.formatNumber(current, 0);
        if (data.ep_reward_mean !== undefined && rewardChart) {
            updateChart(rewardChart, label, data.ep_reward_mean);
        }
        if (data.policy_loss !== undefined && lossChart) {
            updateChart(lossChart, label, data.policy_loss);
        } else if (data.loss !== undefined && lossChart) {
            updateChart(lossChart, label, data.loss);
        }
    });

    // ---- Training Checkpoint Events ----
    socket.on('training_checkpoint', function (data) {
        var msg = 'Checkpoint saved';
        if (data.path) msg += ': ' + data.path;
        if (data.timesteps) msg += ' (step ' + window.formatNumber(data.timesteps, 0) + ')';
        logMessage(msg, 'success');
    });

    // ---- Log Helper ----
    function logMessage(text, type) {
        type = type || '';
        var now = new Date();
        var ts  = [
            String(now.getHours()).padStart(2, '0'),
            String(now.getMinutes()).padStart(2, '0'),
            String(now.getSeconds()).padStart(2, '0')
        ].join(':');

        var entry = document.createElement('div');
        entry.className = 'log-entry' + (type ? ' ' + type : '');
        entry.innerHTML = '<span class="timestamp">[' + ts + ']</span> ' + text;
        trainingLog.appendChild(entry);
        trainingLog.scrollTop = trainingLog.scrollHeight;
    }

    btnClearLog.addEventListener('click', function () {
        trainingLog.innerHTML = '';
        logMessage('Log cleared.', 'info');
    });

})();
