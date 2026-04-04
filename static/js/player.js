/* ============================================================
   DOOM RL – Play / Demo Page JS
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
    var statusBadge     = document.getElementById('demoStatusBadge');
    var gameOverlay     = document.getElementById('gameOverlay');
    var canvas          = document.getElementById('gameCanvas');
    var ctx             = canvas ? canvas.getContext('2d') : null;

    // Stats
    var currentRewardEl = document.getElementById('currentReward');
    var totalRewardEl   = document.getElementById('totalReward');
    var stepCountEl     = document.getElementById('stepCount');
    var episodeCountEl  = document.getElementById('episodeCount');
    var episodeHistory  = document.getElementById('episodeHistory');

    var isPlaying   = false;
    var totalReward = 0;
    var episodeNum  = 0;
    var stepNum     = 0;

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

    // ---- Fetch Models ----
    fetch('/api/models')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            modelSelect.innerHTML = '<option value="">-- Select Model --</option>';
            if (data.models && data.models.length > 0) {
                data.models.forEach(function (m) {
                    var opt = document.createElement('option');
                    opt.value = m.name || m.path;
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

    // ---- Start Demo ----
    btnStart.addEventListener('click', function () {
        var model = modelSelect.value;
        if (!model) {
            window.showToast('Please select a model first.', 'warning');
            return;
        }

        // Reset state
        totalReward = 0;
        episodeNum  = 0;
        stepNum     = 0;
        updateStats(0, 0, 0, 0);
        clearEpisodeHistory();

        socket.emit('start_demo', {
            model:    model,
            scenario: scenarioSelect.value || '',
            speed:    parseFloat(speedSlider.value)
        });

        setPlayingState(true);
        if (gameOverlay) gameOverlay.style.display = 'none';
    });

    // ---- Stop Demo ----
    btnStop.addEventListener('click', function () {
        socket.emit('stop_demo');
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
        if (status === 'running' || status === 'playing') {
            setPlayingState(true);
            if (gameOverlay) gameOverlay.style.display = 'none';
        } else if (status === 'stopped' || status === 'error') {
            setPlayingState(false);
            if (gameOverlay) {
                gameOverlay.style.display = '';
                gameOverlay.textContent = status === 'error'
                    ? 'Error: ' + (data.message || 'Unknown')
                    : 'Demo Stopped';
            }
            if (status === 'error') {
                window.showToast('Demo error: ' + (data.message || ''), 'error');
            }
        }
    });

    // ---- Helpers ----
    function setPlayingState(playing) {
        isPlaying = playing;
        btnStart.disabled = playing;
        btnStop.disabled  = !playing;
        statusBadge.textContent = playing ? 'Playing' : 'Idle';
        statusBadge.className = playing ? 'mono text-success training-active' : 'mono text-muted';
    }

    function updateStats(current, total, step, episode) {
        if (currentRewardEl) currentRewardEl.textContent = window.formatNumber(current);
        if (totalRewardEl)   totalRewardEl.textContent   = window.formatNumber(total);
        if (stepCountEl)     stepCountEl.textContent      = step;
        if (episodeCountEl)  episodeCountEl.textContent   = episode;
    }

    function addEpisodeToHistory(epNum, reward) {
        // Remove empty state on first episode
        if (epNum === 1) {
            episodeHistory.innerHTML = '';
        }

        var item = document.createElement('div');
        item.className = 'episode-item';
        item.innerHTML =
            '<span class="ep-num">Episode ' + epNum + '</span>' +
            '<span class="ep-reward">' + window.formatNumber(reward) + '</span>';

        // Prepend so newest is on top
        episodeHistory.insertBefore(item, episodeHistory.firstChild);
    }

    function clearEpisodeHistory() {
        episodeHistory.innerHTML =
            '<div class="empty-state" style="padding:30px;">' +
            '<i class="fas fa-inbox"></i>' +
            '<p>Episode rewards will appear here as the agent completes episodes.</p>' +
            '</div>';
    }

})();
