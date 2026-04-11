/* ============================================================
   DOOM RL – Training Page JS (Tabbed Sessions)
   ============================================================ */

(function () {
    'use strict';

    var socket = window.socket;

    // ---- DOM References ----
    var scenarioSelect = document.getElementById('scenarioSelect');
    var scenarioDesc   = document.getElementById('scenarioDesc');
    var btnStart       = document.getElementById('btnStart');
    var btnLoadDefaults= document.getElementById('btnLoadDefaults');
    var tabBar         = document.getElementById('sessionTabBar');
    var panelContainer = document.getElementById('sessionPanels');
    var noSessionPanel = document.getElementById('noSessionPanel');

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

    var scenarioDefaults = {};
    var scenarioDescriptions = {};
    var sessions = {};
    var activeTabId = null;
    var pendingConfig = null; // stash config while waiting for "started"

    // ---- User Context ----
    function getCurrentUser() {
        try { return JSON.parse(localStorage.getItem('doom_user')) || null; } catch(e) { return null; }
    }
    function getUsername() {
        var u = getCurrentUser();
        return u ? u.username : 'anonymous';
    }

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

    // ---- Fetch Scenarios & Auto-select First ----
    fetch('/api/scenarios')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            scenarioSelect.innerHTML = '';
            if (data.scenarios && data.scenarios.length > 0) {
                data.scenarios.forEach(function (s) {
                    var opt = document.createElement('option');
                    opt.value = s.id || s.name;
                    opt.textContent = s.name || s.id;
                    scenarioSelect.appendChild(opt);
                    var key = s.id || s.name;
                    if (s.defaults) scenarioDefaults[key] = s.defaults;
                    scenarioDescriptions[key] = s.description || '';
                });
                scenarioSelect.selectedIndex = 0;
                loadDefaultsForScenario(scenarioSelect.value);
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
        scenarioDesc.textContent = scenarioDescriptions[scenarioId] || '';
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

    // ============================================================
    //  BUTTON STATE HELPERS
    // ============================================================

    function setStartButtonLoading(loading) {
        if (loading) {
            btnStart.disabled = true;
            btnStart.classList.add('loading');
            btnStart.innerHTML = '<span class="spinner-inline"></span> Starting...';
        } else {
            btnStart.disabled = false;
            btnStart.classList.remove('loading');
            btnStart.innerHTML = '<i class="fas fa-play"></i> Start Training';
        }
    }

    // ============================================================
    //  SESSION TAB MANAGEMENT
    // ============================================================

    function createSessionTab(sessionId, scenarioName, configSnapshot) {
        noSessionPanel.style.display = 'none';

        // Tab button
        var tab = document.createElement('button');
        tab.className = 'session-tab active animate-slideIn';
        tab.dataset.session = sessionId;
        tab.innerHTML =
            '<span class="session-tab-dot pulse-dot"></span>' +
            '<span class="session-tab-label">' + escHtml(scenarioName) + '</span>' +
            '<span class="session-tab-id">#' + sessionId + '</span>' +
            '<span class="session-tab-status">Starting</span>' +
            '<button class="session-tab-close" title="Close tab">&times;</button>';

        // Deactivate others
        deactivateAllTabs();
        tabBar.appendChild(tab);

        // Panel
        var panel = document.createElement('div');
        panel.className = 'session-panel animate-slideIn';
        panel.id = 'panel-' + sessionId;
        panel.innerHTML = buildPanelHTML(sessionId, scenarioName, configSnapshot);
        panelContainer.appendChild(panel);

        // Initialize charts
        var rCanvas = panel.querySelector('.reward-chart-canvas');
        var lCanvas = panel.querySelector('.loss-chart-canvas');
        var rewardChart = rCanvas ? createChartFromCanvas(rCanvas, 'reward') : null;
        var lossChart = lCanvas ? createChartFromCanvas(lCanvas, 'loss') : null;

        sessions[sessionId] = {
            tab: tab,
            panel: panel,
            rewardChart: rewardChart,
            lossChart: lossChart,
            config: configSnapshot,
            status: 'starting',
            updateCounter: 0,
            startTime: Date.now()
        };

        activeTabId = sessionId;

        // Tab click
        tab.addEventListener('click', function (e) {
            if (e.target.classList.contains('session-tab-close')) return;
            switchToTab(sessionId);
        });
        tab.querySelector('.session-tab-close').addEventListener('click', function (e) {
            e.stopPropagation();
            closeSessionTab(sessionId);
        });

        // Wire panel buttons
        wireSessionButtons(sessionId, panel);

        logToSession(sessionId, 'Session created for ' + scenarioName, 'info');
    }

    function wireSessionButtons(sessionId, panel) {
        var btnPause  = panel.querySelector('.btn-pause');
        var btnResume = panel.querySelector('.btn-resume');
        var btnStop   = panel.querySelector('.btn-stop');
        var btnClear  = panel.querySelector('.btn-clear-log');

        btnPause.addEventListener('click', function () {
            socket.emit('pause_training', { session_id: sessionId });
            logToSession(sessionId, 'Pause requested...', 'info');
        });
        btnResume.addEventListener('click', function () {
            socket.emit('resume_training', { session_id: sessionId });
            logToSession(sessionId, 'Resume requested...', 'info');
        });
        btnStop.addEventListener('click', function () {
            if (confirm('Stop training #' + sessionId + '?')) {
                socket.emit('stop_training', { session_id: sessionId });
                logToSession(sessionId, 'Stop requested...', 'warning');
            }
        });
        btnClear.addEventListener('click', function () {
            panel.querySelector('.training-log').innerHTML = '';
            logToSession(sessionId, 'Log cleared.', 'info');
        });
    }

    function buildPanelHTML(sessionId, scenarioName, cfg) {
        var rewardTags = '';
        if (cfg.kill_reward) rewardTags += '<span class="cfg-tag reward">Kill +' + cfg.kill_reward + '</span>';
        if (cfg.damage_penalty) rewardTags += '<span class="cfg-tag penalty">Dmg ' + cfg.damage_penalty + '</span>';
        if (cfg.miss_penalty) rewardTags += '<span class="cfg-tag penalty">Miss ' + cfg.miss_penalty + '</span>';
        if (cfg.step_penalty) rewardTags += '<span class="cfg-tag penalty">Step ' + cfg.step_penalty + '</span>';
        if (cfg.ammo_penalty) rewardTags += '<span class="cfg-tag penalty">Ammo ' + cfg.ammo_penalty + '</span>';

        return '' +
        '<div class="card mb-3">' +
            '<div class="card-header">' +
                '<h3><i class="fas fa-tasks text-primary"></i> <span class="session-title">' + escHtml(scenarioName) + '</span> <span class="session-id-label">#' + sessionId + '</span></h3>' +
                '<div class="btn-group">' +
                    '<button class="btn btn-sm btn-secondary btn-pause"><i class="fas fa-pause"></i> Pause</button>' +
                    '<button class="btn btn-sm btn-secondary btn-resume" style="display:none;"><i class="fas fa-play"></i> Resume</button>' +
                    '<button class="btn btn-sm btn-danger btn-stop"><i class="fas fa-stop"></i> Stop</button>' +
                '</div>' +
            '</div>' +

            '<div class="session-config-summary">' +
                '<span class="cfg-tag">LR ' + cfg.learning_rate + '</span>' +
                '<span class="cfg-tag">Steps ' + cfg.n_steps + '</span>' +
                '<span class="cfg-tag">Clip ' + cfg.clip_range + '</span>' +
                '<span class="cfg-tag">Gamma ' + cfg.gamma + '</span>' +
                '<span class="cfg-tag">Lambda ' + cfg.gae_lambda + '</span>' +
                '<span class="cfg-tag highlight">Timesteps ' + window.formatNumber(cfg.total_timesteps, 0) + '</span>' +
                rewardTags +
            '</div>' +

            '<div class="session-status-row mt-2">' +
                '<div class="session-status-left">' +
                    '<span class="session-status-badge badge-starting"><i class="fas fa-circle-notch fa-spin"></i> Starting</span>' +
                    '<span class="session-status-msg text-muted" style="font-size:0.8rem;margin-left:12px;"></span>' +
                '</div>' +
                '<div class="session-status-right">' +
                    '<span class="mono text-secondary-c session-timestep" style="font-size:0.85rem;">0 / ' + window.formatNumber(cfg.total_timesteps, 0) + '</span>' +
                '</div>' +
            '</div>' +
            '<div class="progress-bar-wrapper mt-2">' +
                '<div class="progress-bar-fill session-progress"></div>' +
            '</div>' +
            '<div class="session-time-row mt-2">' +
                '<span class="session-elapsed text-muted" style="font-size:0.78rem;"><i class="fas fa-clock"></i> Elapsed: 0s</span>' +
                '<span class="session-eta text-muted" style="font-size:0.78rem;"><i class="fas fa-hourglass-half"></i> ETA: --</span>' +
            '</div>' +

            '<div class="stats-grid mt-3" style="margin-bottom:0;">' +
                '<div class="stat-card"><div class="stat-label">FPS</div><div class="stat-value session-fps" style="font-size:1.4rem;">--</div></div>' +
                '<div class="stat-card"><div class="stat-label">Reward (Mean)</div><div class="stat-value primary session-reward" style="font-size:1.4rem;">--</div></div>' +
                '<div class="stat-card"><div class="stat-label">Ep Length</div><div class="stat-value success session-eplen" style="font-size:1.4rem;">--</div></div>' +
                '<div class="stat-card"><div class="stat-label">Entropy</div><div class="stat-value session-entropy" style="font-size:1.4rem;">--</div></div>' +
                '<div class="stat-card"><div class="stat-label">Episodes</div><div class="stat-value session-episodes" style="font-size:1.4rem;">0</div></div>' +
                '<div class="stat-card"><div class="stat-label">Clip Frac</div><div class="stat-value session-clip" style="font-size:1.4rem;">--</div></div>' +
            '</div>' +
        '</div>' +

        '<div class="card mb-3">' +
            '<div class="card-header"><h3><i class="fas fa-chart-line text-secondary-c"></i> Episode Reward</h3></div>' +
            '<div class="chart-container"><canvas class="reward-chart-canvas"></canvas></div>' +
        '</div>' +
        '<div class="card mb-3">' +
            '<div class="card-header"><h3><i class="fas fa-chart-area text-primary"></i> Policy Loss</h3></div>' +
            '<div class="chart-container"><canvas class="loss-chart-canvas"></canvas></div>' +
        '</div>' +

        '<div class="card">' +
            '<div class="card-header">' +
                '<h3><i class="fas fa-terminal text-muted"></i> Training Log</h3>' +
                '<button class="btn btn-sm btn-outline btn-clear-log"><i class="fas fa-trash"></i> Clear</button>' +
            '</div>' +
            '<div class="training-log"><div class="log-entry info"><span class="timestamp">[--:--:--]</span> Waiting for training to begin...</div></div>' +
        '</div>';
    }

    function deactivateAllTabs() {
        var tabs = tabBar.querySelectorAll('.session-tab');
        for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove('active');
        var panels = panelContainer.querySelectorAll('.session-panel');
        for (var j = 0; j < panels.length; j++) panels[j].style.display = 'none';
    }

    function switchToTab(sessionId) {
        deactivateAllTabs();
        if (sessions[sessionId]) {
            sessions[sessionId].tab.classList.add('active');
            sessions[sessionId].panel.style.display = '';
            activeTabId = sessionId;
        }
    }

    function closeSessionTab(sessionId) {
        var s = sessions[sessionId];
        if (!s) return;
        if (s.rewardChart) s.rewardChart.destroy();
        if (s.lossChart) s.lossChart.destroy();
        s.tab.remove();
        s.panel.remove();
        delete sessions[sessionId];
        var remaining = Object.keys(sessions);
        if (remaining.length > 0) {
            switchToTab(remaining[remaining.length - 1]);
        } else {
            activeTabId = null;
            noSessionPanel.style.display = '';
        }
    }

    // ============================================================
    //  CHART FACTORY
    // ============================================================

    function createChartFromCanvas(canvasEl, type) {
        var ctx = canvasEl.getContext('2d');
        var color = type === 'reward' ? '#00f0ff' : '#ff4400';
        var rgbaHi = type === 'reward' ? 'rgba(0,240,255,0.25)' : 'rgba(255,68,0,0.2)';
        var rgbaLo = type === 'reward' ? 'rgba(0,240,255,0.0)' : 'rgba(255,68,0,0.0)';
        var gradient = ctx.createLinearGradient(0, 0, 0, 280);
        gradient.addColorStop(0, rgbaHi);
        gradient.addColorStop(1, rgbaLo);
        return new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: type === 'reward' ? 'Episode Reward' : 'Policy Loss', data: [], borderColor: color, backgroundColor: gradient, borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0, pointHoverRadius: 4, pointHoverBackgroundColor: color }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { maxTicksLimit: 10, color: '#606070' } },
                    y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#606070' } }
                },
                plugins: { tooltip: { backgroundColor: 'rgba(20,20,35,0.95)', titleColor: '#fff', bodyColor: color, borderColor: color.replace(')', ',0.3)').replace('rgb', 'rgba'), borderWidth: 1, padding: 10 } }
            }
        });
    }

    // ============================================================
    //  HELPERS
    // ============================================================

    function logToSession(sessionId, text, type) {
        var s = sessions[sessionId];
        if (!s) return;
        var log = s.panel.querySelector('.training-log');
        type = type || '';
        var now = new Date();
        var ts = [
            String(now.getHours()).padStart(2, '0'),
            String(now.getMinutes()).padStart(2, '0'),
            String(now.getSeconds()).padStart(2, '0')
        ].join(':');
        var entry = document.createElement('div');
        entry.className = 'log-entry log-fade-in' + (type ? ' ' + type : '');
        entry.innerHTML = '<span class="timestamp">[' + ts + ']</span> ' + text;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;
    }

    function flashValue(el) {
        el.classList.remove('value-flash');
        void el.offsetWidth; // force reflow
        el.classList.add('value-flash');
    }

    function escHtml(str) {
        var d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    function updateSessionBadge(panel, text, cssClass) {
        var badge = panel.querySelector('.session-status-badge');
        badge.className = 'session-status-badge ' + cssClass;
        badge.innerHTML = text;
    }

    function updateSessionMsg(panel, msg) {
        var el = panel.querySelector('.session-status-msg');
        if (el) el.textContent = msg || '';
    }

    // ============================================================
    //  START TRAINING
    // ============================================================

    btnStart.addEventListener('click', function () {
        var scenario = scenarioSelect.value;
        if (!scenario) {
            window.showToast('Please select a scenario first.', 'warning');
            return;
        }

        pendingConfig = {
            scenario:        scenario,
            learning_rate:   parseFloat(inputs.learningRate.value),
            n_steps:         parseInt(inputs.nSteps.value),
            clip_range:      parseFloat(inputs.clipRange.value),
            gamma:           parseFloat(inputs.gamma.value),
            gae_lambda:      parseFloat(inputs.gaeLambda.value),
            total_timesteps: parseInt(inputs.totalTimesteps.value),
            kill_reward:     parseFloat(inputs.killReward.value),
            miss_penalty:    parseFloat(inputs.missPenalty.value),
            step_penalty:    parseFloat(inputs.stepPenalty.value),
            damage_penalty:  parseFloat(inputs.damagePenalty.value),
            ammo_penalty:    parseFloat(inputs.ammoPenalty.value)
        };

        pendingConfig.user = getUsername();
        setStartButtonLoading(true);
        socket.emit('start_training', pendingConfig);
    });

    // ============================================================
    //  SOCKET: training_status
    // ============================================================

    socket.on('training_status', function (data) {
        var sessionId = data.session_id;
        var status = data.status;
        if (!sessionId) return;

        // ---- New session: create tab ----
        if (status === 'started' && !sessions[sessionId]) {
            setStartButtonLoading(false);
            var scenario = data.scenario || (pendingConfig && pendingConfig.scenario) || 'Unknown';
            var selOpt = scenarioSelect.querySelector('option[value="' + scenario + '"]');
            var displayName = selOpt ? selOpt.textContent : scenario;
            var cfg = pendingConfig || { scenario: scenario, learning_rate: 0.0001, n_steps: 2048, clip_range: 0.2, gamma: 0.99, gae_lambda: 0.95, total_timesteps: 100000, kill_reward: 0, miss_penalty: 0, step_penalty: 0, damage_penalty: 0, ammo_penalty: 0 };
            pendingConfig = null;
            createSessionTab(sessionId, displayName, cfg);
            logToSession(sessionId, 'Session started successfully.', 'success');
            return;
        }

        // ---- Failed with no tab: show error & reset button ----
        if ((status === 'failed' || status === 'error') && !sessions[sessionId]) {
            setStartButtonLoading(false);
            pendingConfig = null;
            var errMsg = data.error || data.message || 'Unknown error';
            window.showToast('Training failed: ' + errMsg, 'error');
            return;
        }

        var s = sessions[sessionId];
        if (!s) return;

        s.status = status;
        var panel = s.panel;
        var tabDot = s.tab.querySelector('.session-tab-dot');
        var tabStatus = s.tab.querySelector('.session-tab-status');
        var btnPause = panel.querySelector('.btn-pause');
        var btnResume = panel.querySelector('.btn-resume');
        var btnStop = panel.querySelector('.btn-stop');

        tabStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);

        if (status === 'initializing') {
            tabDot.className = 'session-tab-dot pulse-dot';
            tabDot.style.background = 'var(--warning)';
            updateSessionBadge(panel, '<i class="fas fa-cog fa-spin"></i> Initializing', 'badge-init');
            updateSessionMsg(panel, data.message || 'Setting up...');
            logToSession(sessionId, data.message || 'Initializing...', 'info');

        } else if (status === 'training' || status === 'running') {
            tabDot.className = 'session-tab-dot pulse-dot';
            tabDot.style.background = 'var(--success)';
            updateSessionBadge(panel, '<i class="fas fa-running"></i> Training', 'badge-running');
            updateSessionMsg(panel, '');
            btnPause.style.display = '';
            btnResume.style.display = 'none';
            btnPause.disabled = false;
            btnStop.disabled = false;
            logToSession(sessionId, 'Training in progress...', 'success');

        } else if (status === 'paused') {
            tabDot.className = 'session-tab-dot';
            tabDot.style.background = 'var(--warning)';
            updateSessionBadge(panel, '<i class="fas fa-pause"></i> Paused', 'badge-paused');
            btnPause.style.display = 'none';
            btnResume.style.display = '';
            btnResume.disabled = false;
            logToSession(sessionId, 'Training paused.', 'info');

        } else if (status === 'completed') {
            tabDot.className = 'session-tab-dot';
            tabDot.style.background = 'var(--success)';
            updateSessionBadge(panel, '<i class="fas fa-check-circle"></i> Completed', 'badge-completed');
            btnPause.disabled = true;
            btnResume.style.display = 'none';
            btnStop.disabled = true;
            logToSession(sessionId, 'Training completed! Model saved.', 'success');
            window.showToast('Session #' + sessionId + ' completed!', 'success');

        } else if (status === 'stopped') {
            tabDot.className = 'session-tab-dot';
            tabDot.style.background = 'var(--text-muted)';
            updateSessionBadge(panel, '<i class="fas fa-stop-circle"></i> Stopped', 'badge-stopped');
            btnPause.disabled = true;
            btnResume.style.display = 'none';
            btnStop.disabled = true;
            logToSession(sessionId, 'Training stopped by user.', 'warning');

        } else if (status === 'failed' || status === 'error') {
            tabDot.className = 'session-tab-dot';
            tabDot.style.background = 'var(--danger)';
            var errMsg = data.error || data.message || 'Unknown error';
            updateSessionBadge(panel, '<i class="fas fa-exclamation-triangle"></i> Failed', 'badge-error');
            updateSessionMsg(panel, errMsg);
            btnPause.disabled = true;
            btnResume.style.display = 'none';
            btnStop.disabled = true;
            logToSession(sessionId, 'Error: ' + errMsg, 'error');
            window.showToast('Session #' + sessionId + ' failed: ' + errMsg, 'error');
        }
    });

    // ============================================================
    //  SOCKET: training_metrics
    // ============================================================

    socket.on('training_metrics', function (data) {
        var sessionId = data.session_id;
        var s = sessionId ? sessions[sessionId] : null;

        // Fallback: find most recent active session
        if (!s) {
            var keys = Object.keys(sessions);
            for (var k = keys.length - 1; k >= 0; k--) {
                var st = sessions[keys[k]].status;
                if (st === 'training' || st === 'running' || st === 'starting') {
                    s = sessions[keys[k]];
                    sessionId = keys[k];
                    break;
                }
            }
        }
        if (!s) return;

        s.updateCounter++;
        var panel = s.panel;
        var total = data.total_timesteps || s.config.total_timesteps || 1;
        var current = data.timesteps || data.timestep || data.total_timesteps_done || 0;
        var pct = Math.min((current / total) * 100, 100);

        // Progress bar
        var progBar = panel.querySelector('.session-progress');
        progBar.style.width = pct.toFixed(1) + '%';
        if (!progBar.classList.contains('active')) progBar.classList.add('active');

        // Timestep counter
        panel.querySelector('.session-timestep').textContent =
            window.formatNumber(current, 0) + ' / ' + window.formatNumber(total, 0);

        // Elapsed & ETA
        var elapsed = data.elapsed_time || 0;
        var elapsedEl = panel.querySelector('.session-elapsed');
        var etaEl = panel.querySelector('.session-eta');
        elapsedEl.innerHTML = '<i class="fas fa-clock"></i> Elapsed: ' + window.formatTime(elapsed);
        if (data.fps && data.fps > 0 && current < total) {
            var remaining = (total - current) / data.fps;
            etaEl.innerHTML = '<i class="fas fa-hourglass-half"></i> ETA: ' + window.formatTime(remaining);
        } else if (current >= total) {
            etaEl.innerHTML = '<i class="fas fa-check"></i> Done';
        }

        // Stats with flash animation
        if (data.fps !== undefined) {
            var fpsEl = panel.querySelector('.session-fps');
            fpsEl.textContent = Math.round(data.fps);
            flashValue(fpsEl);
        }
        if (data.ep_reward_mean !== undefined) {
            var rwEl = panel.querySelector('.session-reward');
            rwEl.textContent = window.formatNumber(data.ep_reward_mean);
            flashValue(rwEl);
        }
        if (data.ep_len_mean !== undefined) {
            var lenEl = panel.querySelector('.session-eplen');
            lenEl.textContent = window.formatNumber(data.ep_len_mean, 0);
            flashValue(lenEl);
        }
        if (data.entropy_loss !== undefined) {
            var entEl = panel.querySelector('.session-entropy');
            entEl.textContent = window.formatNumber(data.entropy_loss, 4);
            flashValue(entEl);
        }
        if (data.total_episodes !== undefined) {
            panel.querySelector('.session-episodes').textContent = data.total_episodes;
        }
        if (data.clip_fraction !== undefined) {
            panel.querySelector('.session-clip').textContent = window.formatNumber(data.clip_fraction, 4);
        }

        // Charts
        var label = window.formatNumber(current, 0);
        if (data.ep_reward_mean !== undefined && s.rewardChart) {
            updateChart(s.rewardChart, label, data.ep_reward_mean);
        }
        if (data.policy_loss !== undefined && s.lossChart) {
            updateChart(s.lossChart, label, data.policy_loss);
        } else if (data.loss !== undefined && s.lossChart) {
            updateChart(s.lossChart, label, data.loss);
        }
    });

    // ============================================================
    //  SOCKET: training_checkpoint
    // ============================================================

    socket.on('training_checkpoint', function (data) {
        var sessionId = data.session_id;
        if (!sessionId || !sessions[sessionId]) {
            var keys = Object.keys(sessions);
            if (keys.length > 0) sessionId = keys[keys.length - 1];
            else return;
        }
        var msg = '<i class="fas fa-save"></i> Checkpoint saved';
        if (data.timesteps) msg += ' at step ' + window.formatNumber(data.timesteps, 0);
        logToSession(sessionId, msg, 'success');
    });

    // ============================================================
    //  LOAD TRAINING HISTORY ON PAGE LOAD
    // ============================================================

    function loadTrainingHistory() {
        var user = getUsername();
        fetch('/api/training/history?user=' + encodeURIComponent(user))
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (!data.sessions || data.sessions.length === 0) return;
                // Show completed/failed sessions as history tabs
                data.sessions.forEach(function(s) {
                    if (sessions[s.session_id]) return; // already open
                    if (s.status !== 'completed' && s.status !== 'failed' && s.status !== 'stopped') return;
                    createHistoryTab(s);
                });
            })
            .catch(function() {});
    }

    function createHistoryTab(historySession) {
        noSessionPanel.style.display = 'none';

        var sid = historySession.session_id;
        var scenario = historySession.scenario || 'Unknown';
        var displayName = scenario.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
        var status = historySession.status;

        // Tab
        var tab = document.createElement('button');
        tab.className = 'session-tab';
        tab.dataset.session = sid;
        var dotColor = status === 'completed' ? 'var(--success)' : status === 'failed' ? 'var(--danger)' : 'var(--text-muted)';
        tab.innerHTML =
            '<span class="session-tab-dot" style="background:' + dotColor + '"></span>' +
            '<span class="session-tab-label">' + escHtml(displayName) + '</span>' +
            '<span class="session-tab-id">#' + sid + '</span>' +
            '<span class="session-tab-status">' + status.charAt(0).toUpperCase() + status.slice(1) + '</span>' +
            '<button class="session-tab-close" title="Close tab">&times;</button>';

        tabBar.appendChild(tab);

        // Panel
        var panel = document.createElement('div');
        panel.className = 'session-panel';
        panel.id = 'panel-' + sid;
        panel.style.display = 'none';

        var badgeClass = status === 'completed' ? 'badge-completed' : status === 'failed' ? 'badge-error' : 'badge-stopped';
        var badgeIcon = status === 'completed' ? 'check-circle' : status === 'failed' ? 'exclamation-triangle' : 'stop-circle';
        var startedAt = historySession.started_at ? new Date(historySession.started_at).toLocaleString() : '--';
        var finishedAt = historySession.finished_at ? new Date(historySession.finished_at).toLocaleString() : '--';
        var hp = historySession.hyperparams || {};
        var rw = historySession.reward_config || {};

        panel.innerHTML =
            '<div class="card mb-3">' +
                '<div class="card-header">' +
                    '<h3><i class="fas fa-history text-primary"></i> <span class="session-title">' + escHtml(displayName) + '</span> <span class="session-id-label">#' + sid + '</span></h3>' +
                    '<span class="session-status-badge ' + badgeClass + '"><i class="fas fa-' + badgeIcon + '"></i> ' + status.charAt(0).toUpperCase() + status.slice(1) + '</span>' +
                '</div>' +
                '<div class="session-config-summary">' +
                    (hp.learning_rate ? '<span class="cfg-tag">LR ' + hp.learning_rate + '</span>' : '') +
                    (hp.n_steps ? '<span class="cfg-tag">Steps ' + hp.n_steps + '</span>' : '') +
                    (hp.gamma ? '<span class="cfg-tag">Gamma ' + hp.gamma + '</span>' : '') +
                    '<span class="cfg-tag highlight">Timesteps ' + window.formatNumber(historySession.total_timesteps || 0, 0) + '</span>' +
                    (rw.kill_reward ? '<span class="cfg-tag reward">Kill +' + rw.kill_reward + '</span>' : '') +
                    (rw.damage_penalty ? '<span class="cfg-tag penalty">Dmg ' + rw.damage_penalty + '</span>' : '') +
                '</div>' +
                '<div class="stats-grid mt-3" style="margin-bottom:0;">' +
                    '<div class="stat-card"><div class="stat-label">Mean Reward</div><div class="stat-value primary" style="font-size:1.4rem;">' + (historySession.mean_reward != null ? historySession.mean_reward : '--') + '</div></div>' +
                    '<div class="stat-card"><div class="stat-label">Episodes</div><div class="stat-value" style="font-size:1.4rem;">' + (historySession.total_episodes || 0) + '</div></div>' +
                    '<div class="stat-card"><div class="stat-label">Started</div><div class="stat-value" style="font-size:0.85rem;">' + startedAt + '</div></div>' +
                    '<div class="stat-card"><div class="stat-label">Finished</div><div class="stat-value" style="font-size:0.85rem;">' + finishedAt + '</div></div>' +
                '</div>' +
                (historySession.error ? '<div class="mt-2" style="color:var(--danger);font-size:0.85rem;"><i class="fas fa-exclamation-triangle"></i> ' + escHtml(historySession.error) + '</div>' : '') +
            '</div>';

        panelContainer.appendChild(panel);

        sessions[sid] = { tab: tab, panel: panel, rewardChart: null, lossChart: null, config: hp, status: status, updateCounter: 0, startTime: 0, isHistory: true };

        tab.addEventListener('click', function(e) {
            if (e.target.classList.contains('session-tab-close')) return;
            switchToTab(sid);
        });
        tab.querySelector('.session-tab-close').addEventListener('click', function(e) {
            e.stopPropagation();
            closeSessionTab(sid);
        });
    }

    // Load history on page ready
    loadTrainingHistory();

})();
