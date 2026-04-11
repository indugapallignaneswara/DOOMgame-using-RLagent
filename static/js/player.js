/* ============================================================
   ARENA OF RL - Play / Demo Page JS
   ============================================================ */

(function () {
    'use strict';

    var socket = window.socket;

    // ---- DOM References ----
    var modelSelect     = document.getElementById('modelSelect');
    var scenarioSelect  = document.getElementById('playScenarioSelect');
    var speedSlider     = document.getElementById('speedSlider');
    var speedVal        = document.getElementById('speedVal');
    var btnStart        = document.getElementById('btnStartDemo');
    var btnStop         = document.getElementById('btnStopDemo');
    var btnFullscreen   = document.getElementById('btnFullscreen');
    var statusBadge     = document.getElementById('demoStatusBadge');
    var gameOverlay     = document.getElementById('gameOverlay');
    var gameContainer   = document.getElementById('gameContainer');
    var canvas          = document.getElementById('gameCanvas');
    var ctx             = canvas ? canvas.getContext('2d') : null;
    var actionOverlay   = document.getElementById('actionOverlay');
    var hudOverlay      = document.getElementById('hudOverlay');
    var scanlineToggle  = document.getElementById('scanlineToggle');

    // Stats
    var currentRewardEl = document.getElementById('currentReward');
    var totalRewardEl   = document.getElementById('totalReward');
    var stepCountEl     = document.getElementById('stepCount');
    var episodeCountEl  = document.getElementById('episodeCount');
    var episodeHistory  = document.getElementById('episodeHistory');

    // HUD elements
    var hudHealth       = document.getElementById('hudHealthFill');
    var hudHealthText   = document.getElementById('hudHealthText');
    var hudAmmo         = document.getElementById('hudAmmoText');
    var hudKills        = document.getElementById('hudKillsText');

    var isPlaying   = false;
    var totalReward = 0;
    var episodeNum  = 0;
    var stepNum     = 0;
    var currentDemoSessionId = null;
    var currentButtons = [];

    // Preloaded image for frame rendering
    var frameImage = new Image();

    // ---- Speed Slider ----
    if (speedSlider) {
        speedSlider.addEventListener('input', function () {
            speedVal.textContent = parseFloat(this.value).toFixed(2) + 's';
            if (isPlaying) {
                socket.emit('set_demo_speed', { speed: parseFloat(this.value) });
            }
        });
    }

    // ---- Scanline Toggle ----
    if (scanlineToggle) {
        scanlineToggle.addEventListener('change', function () {
            var scanlines = document.getElementById('scanlineOverlay');
            if (scanlines) {
                scanlines.style.display = this.checked ? 'block' : 'none';
            }
        });
    }

    // ---- Fullscreen ----
    if (btnFullscreen) {
        btnFullscreen.addEventListener('click', function () {
            var wrapper = document.getElementById('fullscreenWrapper');
            if (wrapper) {
                wrapper.classList.add('active');
                document.body.style.overflow = 'hidden';
            }
        });
    }

    // Exit fullscreen
    document.addEventListener('click', function (e) {
        if (e.target.id === 'fullscreenWrapper' || e.target.id === 'btnExitFullscreen') {
            var wrapper = document.getElementById('fullscreenWrapper');
            if (wrapper) {
                wrapper.classList.remove('active');
                document.body.style.overflow = '';
            }
        }
    });
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            var wrapper = document.getElementById('fullscreenWrapper');
            if (wrapper && wrapper.classList.contains('active')) {
                wrapper.classList.remove('active');
                document.body.style.overflow = '';
            }
        }
    });

    // ---- Fetch Models (BUG-001 FIX: use m.id as value) ----
    fetch('/api/models')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            modelSelect.innerHTML = '<option value="">-- Select Model --</option>';
            if (data.models && data.models.length > 0) {
                data.models.forEach(function (m) {
                    var opt = document.createElement('option');
                    opt.value = m.id;  // BUG-001 FIX: use id, not name
                    opt.textContent = m.name + (m.scenario ? ' (' + m.scenario + ')' : '');
                    opt.dataset.scenario = m.scenario || '';
                    modelSelect.appendChild(opt);
                });
            } else {
                modelSelect.innerHTML = '<option value="">No models available</option>';
            }
        })
        .catch(function () {
            modelSelect.innerHTML = '<option value="">Failed to load models</option>';
        });

    // ---- Auto-fill scenario on model change ----
    modelSelect.addEventListener('change', function () {
        var selected = modelSelect.options[modelSelect.selectedIndex];
        if (selected && selected.dataset.scenario) {
            scenarioSelect.innerHTML = '<option value="' + selected.dataset.scenario + '">' + selected.dataset.scenario + '</option>';
        } else {
            scenarioSelect.innerHTML = '<option value="">Auto-detected from model</option>';
        }
    });

    // ---- Start Demo (BUG-001 FIX: send model_id) ----
    btnStart.addEventListener('click', function () {
        var modelId = modelSelect.value;
        if (!modelId) {
            window.showToast('Please select a model first.', 'warning');
            return;
        }

        // Reset state
        totalReward = 0;
        episodeNum  = 0;
        stepNum     = 0;
        updateStats(0, 0, 0, 0);
        clearEpisodeHistory();
        clearActionOverlay();
        updateHUD(100, 0, 0);

        socket.emit('start_demo', {
            model_id: modelId,  // BUG-001 FIX: send as model_id
            scenario: scenarioSelect.value || '',
            speed:    parseFloat(speedSlider.value)
        });

        setPlayingState(true);
        if (gameOverlay) gameOverlay.style.display = 'none';
    });

    // ---- Stop Demo ----
    btnStop.addEventListener('click', function () {
        socket.emit('stop_demo', { session_id: currentDemoSessionId });
        setPlayingState(false);
    });

    // ---- Game Frame Event ----
    socket.on('game_frame', function (data) {
        if (!canvas || !ctx) return;

        // data.frame is a base64-encoded JPEG/PNG
        if (data.frame) {
            frameImage.onload = function () {
                ctx.drawImage(frameImage, 0, 0, canvas.width, canvas.height);
            };
            frameImage.src = 'data:image/jpeg;base64,' + data.frame;
        }

        // Update live stats
        stepNum++;
        var reward = data.reward || 0;
        totalReward += reward;

        updateStats(reward, totalReward, stepNum, episodeNum);

        // Update action overlay
        if (data.buttons && data.buttons.length > 0) {
            currentButtons = data.buttons;
        }
        if (currentButtons.length > 0 && data.action !== undefined) {
            renderActionOverlay(currentButtons, data.action, data.action_name);
        }

        // Update HUD overlay with game variables
        if (data.info) {
            var health = data.info.health !== undefined ? data.info.health : 100;
            var ammo = data.info.ammo2 !== undefined ? data.info.ammo2 :
                       (data.info.selected_weapon_ammo !== undefined ? data.info.selected_weapon_ammo : 0);
            var kills = data.info.hitcount !== undefined ? data.info.hitcount : 0;
            updateHUD(health, ammo, kills);
        }

        // Check if episode ended
        if (data.done || data.episode_done) {
            episodeNum++;
            addEpisodeToHistory(episodeNum, totalReward);
            totalReward = 0;
            stepNum = 0;
        }
    });

    // ---- Demo Status Event ----
    socket.on('demo_status', function (data) {
        var status = data.status;
        if (data.session_id) currentDemoSessionId = data.session_id;

        // BUG-011 FIX: handle "started" status
        if (status === 'started' || status === 'running' || status === 'playing') {
            setPlayingState(true);
            if (gameOverlay) gameOverlay.style.display = 'none';
        } else if (status === 'stopped' || status === 'error' || status === 'failed') {
            setPlayingState(false);
            if (gameOverlay) {
                gameOverlay.style.display = '';
                // BUG-007 FIX: read data.error, not data.message
                gameOverlay.textContent = (status === 'error' || status === 'failed')
                    ? 'Error: ' + (data.error || 'Unknown')
                    : 'Demo Stopped';
            }
            if (status === 'error' || status === 'failed') {
                // BUG-007 FIX: read data.error
                window.showToast('Demo error: ' + (data.error || ''), 'error');
            }
        }
    });

    // ---- Helpers ----
    function setPlayingState(playing) {
        isPlaying = playing;
        btnStart.disabled = playing;
        btnStop.disabled  = !playing;
        statusBadge.textContent = playing ? 'LIVE' : 'Idle';
        statusBadge.className = playing
            ? 'status-badge-live'
            : 'mono text-muted';
    }

    function updateStats(current, total, step, episode) {
        if (currentRewardEl) currentRewardEl.textContent = window.formatNumber(current);
        if (totalRewardEl)   totalRewardEl.textContent   = window.formatNumber(total);
        if (stepCountEl)     stepCountEl.textContent      = step;
        if (episodeCountEl)  episodeCountEl.textContent   = episode;
    }

    function addEpisodeToHistory(epNum, reward) {
        if (epNum === 1) {
            episodeHistory.innerHTML = '';
        }

        var item = document.createElement('div');
        item.className = 'episode-item animate-slideIn';
        var rewardClass = reward >= 0 ? 'text-success' : 'text-danger';
        item.innerHTML =
            '<span class="ep-num">EP ' + epNum + '</span>' +
            '<span class="ep-reward ' + rewardClass + '">' + window.formatNumber(reward) + '</span>';

        episodeHistory.insertBefore(item, episodeHistory.firstChild);
    }

    function clearEpisodeHistory() {
        episodeHistory.innerHTML =
            '<div class="empty-state" style="padding:30px;">' +
            '<i class="fas fa-inbox"></i>' +
            '<p>Episode rewards will appear here as the agent completes episodes.</p>' +
            '</div>';
    }

    // ---- Action Overlay ----
    function renderActionOverlay(buttons, activeIdx, actionName) {
        if (!actionOverlay) return;
        var html = '';
        for (var i = 0; i < buttons.length; i++) {
            var isActive = (i === activeIdx);
            var cls = 'action-key' + (isActive ? ' active' : '');
            html += '<div class="' + cls + '">' +
                '<span class="action-key-label">' + formatButtonName(buttons[i]) + '</span>' +
                '</div>';
        }
        actionOverlay.innerHTML = html;
    }

    function clearActionOverlay() {
        if (actionOverlay) actionOverlay.innerHTML = '';
    }

    function formatButtonName(name) {
        return name.replace('MOVE_', '').replace('TURN_', 'T-');
    }

    // ---- HUD Overlay ----
    function updateHUD(health, ammo, kills) {
        if (!hudOverlay) return;
        health = Math.max(0, Math.min(100, health));

        if (hudHealth) {
            hudHealth.style.width = health + '%';
            // Color transition: green -> yellow -> red
            if (health > 60) {
                hudHealth.style.background = 'linear-gradient(90deg, #00ff9d, #00cc7d)';
            } else if (health > 30) {
                hudHealth.style.background = 'linear-gradient(90deg, #ffaa00, #ff8800)';
            } else {
                hudHealth.style.background = 'linear-gradient(90deg, #ff2244, #cc1133)';
            }
        }
        if (hudHealthText) hudHealthText.textContent = Math.round(health);
        if (hudAmmo) hudAmmo.textContent = Math.round(ammo);
        if (hudKills) hudKills.textContent = Math.round(kills);
    }

})();
