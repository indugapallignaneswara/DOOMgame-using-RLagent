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
    var saliencyCanvas  = document.getElementById('saliencyCanvas');
    var saliencyCtx     = saliencyCanvas ? saliencyCanvas.getContext('2d') : null;
    var saliencyMode    = document.getElementById('saliencyMode');
    var saliencyModeBadge = document.getElementById('saliencyModeBadge');
    var saliencyOpacity = document.getElementById('saliencyOpacity');
    var saliencyOpacityVal = document.getElementById('saliencyOpacityVal');
    var saliencyOpacityRow = document.getElementById('saliencyOpacityRow');
    var actionOverlay   = document.getElementById('actionOverlay');
    var actionProbs     = document.getElementById('actionProbs');
    var hudOverlay      = document.getElementById('hudOverlay');
    var scanlineToggle  = document.getElementById('scanlineToggle');

    // Belief Inspector
    var predictionToggle = document.getElementById('predictionToggle');
    var predictionEvery  = document.getElementById('predictionEvery');
    var predictionScoreBadge = document.getElementById('predictionScoreBadge');
    var predictionScore  = document.getElementById('predictionScore');
    var predictionWidget = document.getElementById('predictionWidget');
    var predictionPrompt = document.getElementById('predictionPrompt');
    var predictionButtons = document.getElementById('predictionButtons');
    var predictionReveal = document.getElementById('predictionReveal');
    var valueSparkline   = document.getElementById('valueSparkline');
    var valueCurrentEl   = document.getElementById('valueCurrent');
    var valueSparkCtx    = valueSparkline ? valueSparkline.getContext('2d') : null;
    var valueHistory     = [];
    var VALUE_HISTORY_MAX = 200;
    var awaitingPrediction = false;

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

    // Saliency overlay — keep last received heatmap so we can re-draw it on
    // every frame even when the server only sends a new one every K steps.
    var lastSaliencyImage = null;

    // ---- Speed Slider ----
    if (speedSlider) {
        speedSlider.addEventListener('input', function () {
            speedVal.textContent = parseFloat(this.value).toFixed(2) + 's';
            if (isPlaying) {
                socket.emit('set_demo_speed', { speed: parseFloat(this.value) });
            }
        });
    }

    // ---- Saliency mode dropdown ----
    if (saliencyMode) {
        saliencyMode.addEventListener('change', function () {
            var mode = this.value;
            if (saliencyModeBadge) saliencyModeBadge.textContent = mode;
            // Show opacity slider only when overlay is on
            if (saliencyOpacityRow) {
                saliencyOpacityRow.style.display = (mode === 'off') ? 'none' : '';
            }
            // If we're switching off, clear the overlay immediately
            if (mode === 'off' && saliencyCtx) {
                saliencyCtx.clearRect(0, 0, saliencyCanvas.width, saliencyCanvas.height);
                if (saliencyCanvas) saliencyCanvas.style.opacity = '0';
                lastSaliencyImage = null;
            } else if (saliencyCanvas) {
                // Re-apply current opacity setting
                var op = saliencyOpacity ? (parseInt(saliencyOpacity.value, 10) / 100) : 0.6;
                saliencyCanvas.style.opacity = op.toString();
            }
            // Tell the server (only if a demo is running)
            if (isPlaying && currentDemoSessionId) {
                socket.emit('set_saliency_mode', {
                    session_id: currentDemoSessionId,
                    mode: mode
                });
            }
        });
    }

    // ---- Saliency opacity slider (client-side only) ----
    if (saliencyOpacity) {
        saliencyOpacity.addEventListener('input', function () {
            var pct = parseInt(this.value, 10);
            if (saliencyOpacityVal) saliencyOpacityVal.textContent = pct + '%';
            if (saliencyCanvas && saliencyMode && saliencyMode.value !== 'off') {
                saliencyCanvas.style.opacity = (pct / 100).toString();
            }
        });
    }

    // ---- Prediction-mode toggle ----
    if (predictionToggle) {
        predictionToggle.addEventListener('change', function () {
            var enabled = this.checked;
            if (isPlaying && currentDemoSessionId) {
                socket.emit('set_prediction_mode', {
                    session_id: currentDemoSessionId,
                    enabled: enabled,
                    every: predictionEvery ? parseInt(predictionEvery.value, 10) : 8
                });
            }
            if (!enabled) {
                hidePredictionWidget();
            }
        });
    }
    if (predictionEvery) {
        predictionEvery.addEventListener('change', function () {
            // The interval only takes effect when prediction mode is on; if it
            // is, push the new cadence to the running demo immediately.
            if (predictionToggle && predictionToggle.checked &&
                isPlaying && currentDemoSessionId) {
                socket.emit('set_prediction_mode', {
                    session_id: currentDemoSessionId,
                    enabled: true,
                    every: parseInt(predictionEvery.value, 10)
                });
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

        var sMode = saliencyMode ? saliencyMode.value : 'off';
        var pMode = predictionToggle ? predictionToggle.checked : false;
        var pEvery = predictionEvery ? parseInt(predictionEvery.value, 10) : 8;
        socket.emit('start_demo', {
            model_id: modelId,  // BUG-001 FIX: send as model_id
            scenario: scenarioSelect.value || '',
            speed:    parseFloat(speedSlider.value),
            saliency_mode: sMode,
            prediction_mode: pMode,
            prediction_every: pEvery
        });
        // Reset belief-inspector state
        valueHistory = [];
        clearActionProbs();
        hidePredictionWidget();
        updatePredictionScore(0, 0);
        drawValueSparkline();
        // Sync overlay opacity with the slider's current value
        if (saliencyCanvas && sMode !== 'off' && saliencyOpacity) {
            saliencyCanvas.style.opacity = (parseInt(saliencyOpacity.value, 10) / 100).toString();
        }

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

        // ---- Saliency overlay (Brain-cam) ----
        // The server only sends a new saliency PNG every K frames. We cache
        // the last one and re-draw it on every frame so the overlay tracks
        // the game frame visually instead of flickering on/off.
        if (data.saliency_png) {
            var img = new Image();
            img.onload = function () {
                lastSaliencyImage = img;
                if (saliencyCtx && saliencyCanvas && saliencyMode &&
                    saliencyMode.value !== 'off') {
                    saliencyCtx.clearRect(0, 0,
                        saliencyCanvas.width, saliencyCanvas.height);
                    saliencyCtx.drawImage(img, 0, 0,
                        saliencyCanvas.width, saliencyCanvas.height);
                }
            };
            img.src = 'data:image/png;base64,' + data.saliency_png;
        } else if (lastSaliencyImage && saliencyCtx && saliencyCanvas &&
                   saliencyMode && saliencyMode.value !== 'off') {
            // No new saliency this frame — keep showing the last one.
            // (The clearRect+drawImage pattern is cheap on a 640x480 canvas.)
            saliencyCtx.clearRect(0, 0,
                saliencyCanvas.width, saliencyCanvas.height);
            saliencyCtx.drawImage(lastSaliencyImage, 0, 0,
                saliencyCanvas.width, saliencyCanvas.height);
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

        // Belief Inspector — only render bars when the prediction widget
        // isn't holding the screen (Kapoor: no overlay during the freeze).
        if (!awaitingPrediction && data.action_probs && currentButtons.length > 0) {
            renderActionProbs(currentButtons, data.action_probs, data.action);
        }
        if (typeof data.value === 'number') {
            valueHistory.push(data.value);
            if (valueHistory.length > VALUE_HISTORY_MAX) {
                valueHistory.shift();
            }
            if (valueCurrentEl) valueCurrentEl.textContent = data.value.toFixed(2);
            drawValueSparkline();
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

    // ---- Prediction-before-reveal events (Kapoor) ----
    socket.on('prediction_request', function (data) {
        if (!data || !data.top3_actions) return;
        if (data.session_id) currentDemoSessionId = data.session_id;
        // Server already paints a fresh frame for us inside the request — paint
        // it here so the user sees the *current* state without the action overlay.
        if (data.frame && ctx) {
            var img = new Image();
            img.onload = function () {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = 'data:image/jpeg;base64,' + data.frame;
        }
        showPredictionWidget(data.top3_actions, data.buttons || currentButtons,
                             data.freeze_ms || 800);
    });

    socket.on('prediction_result', function (data) {
        if (!data) return;
        revealPrediction(data);
        updatePredictionScore(data.correct_count, data.total_count);
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
            hidePredictionWidget();
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

    // ---- Belief Inspector: action-probability bars ----
    function renderActionProbs(buttons, probs, activeIdx) {
        if (!actionProbs) return;
        var n = Math.min(buttons.length, probs.length);
        var rows = '';
        for (var i = 0; i < n; i++) {
            var p = probs[i] || 0;
            var pct = Math.round(p * 1000) / 10;
            var cls = 'action-prob-row' + (i === activeIdx ? ' active' : '');
            rows +=
                '<div class="' + cls + '">' +
                  '<div class="ap-label">' + formatButtonName(buttons[i]) + '</div>' +
                  '<div class="ap-track"><div class="ap-fill" style="width:' +
                    (p * 100).toFixed(1) + '%"></div></div>' +
                  '<div class="ap-pct">' + pct.toFixed(1) + '%</div>' +
                '</div>';
        }
        actionProbs.innerHTML = rows;
    }
    function clearActionProbs() {
        if (actionProbs) actionProbs.innerHTML = '';
    }

    // ---- Belief Inspector: V(s) sparkline ----
    // Hand-rolled because uPlot isn't bundled. Min-max auto-scaled per draw.
    function drawValueSparkline() {
        if (!valueSparkCtx || !valueSparkline) return;
        var w = valueSparkline.width;
        var h = valueSparkline.height;
        valueSparkCtx.clearRect(0, 0, w, h);
        if (valueHistory.length < 2) return;

        var lo = Infinity, hi = -Infinity;
        for (var i = 0; i < valueHistory.length; i++) {
            var v = valueHistory[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        if (hi - lo < 1e-6) { hi = lo + 1; lo = lo - 1; }
        var pad = 4;
        var plotH = h - pad * 2;
        var plotW = w - pad * 2;
        var n = valueHistory.length;

        // Zero baseline (V(s) crossing 0 is meaningful; show it if in range).
        if (lo < 0 && hi > 0) {
            var zy = pad + plotH * (1 - (0 - lo) / (hi - lo));
            valueSparkCtx.strokeStyle = 'rgba(255,255,255,0.08)';
            valueSparkCtx.setLineDash([3, 3]);
            valueSparkCtx.beginPath();
            valueSparkCtx.moveTo(pad, zy);
            valueSparkCtx.lineTo(w - pad, zy);
            valueSparkCtx.stroke();
            valueSparkCtx.setLineDash([]);
        }

        valueSparkCtx.strokeStyle = '#00f0ff';
        valueSparkCtx.lineWidth = 1.5;
        valueSparkCtx.beginPath();
        for (var j = 0; j < n; j++) {
            var x = pad + (n === 1 ? 0 : (j / (n - 1)) * plotW);
            var y = pad + plotH * (1 - (valueHistory[j] - lo) / (hi - lo));
            if (j === 0) valueSparkCtx.moveTo(x, y);
            else valueSparkCtx.lineTo(x, y);
        }
        valueSparkCtx.stroke();

        // Latest point dot for emphasis
        var lx = pad + plotW;
        var ly = pad + plotH * (1 - (valueHistory[n - 1] - lo) / (hi - lo));
        valueSparkCtx.fillStyle = '#00f0ff';
        valueSparkCtx.beginPath();
        valueSparkCtx.arc(lx, ly, 2.5, 0, Math.PI * 2);
        valueSparkCtx.fill();
    }

    // ---- Prediction-before-reveal widget ----
    function showPredictionWidget(top3, buttons, freezeMs) {
        if (!predictionWidget) return;
        awaitingPrediction = true;
        // Hide overlays during the freeze (Kapoor: no policy info before guess).
        clearActionProbs();
        if (saliencyCanvas) saliencyCanvas.style.opacity = '0';
        if (predictionReveal) {
            predictionReveal.style.display = 'none';
            predictionReveal.textContent = '';
        }
        predictionWidget.style.display = 'flex';
        predictionPrompt.textContent = 'What action will the agent take?';
        predictionButtons.innerHTML = '';

        // Hold the buttons disabled for `freezeMs` so the user studies the
        // unaltered frame first, then the buttons activate.
        var btnEls = [];
        top3.forEach(function (idx) {
            var b = document.createElement('button');
            b.className = 'prediction-btn disabled';
            b.dataset.choice = String(idx);
            b.textContent = buttons[idx] !== undefined
                ? formatButtonName(buttons[idx])
                : ('ACTION ' + idx);
            predictionButtons.appendChild(b);
            btnEls.push(b);
        });
        var other = document.createElement('button');
        other.className = 'prediction-btn other disabled';
        other.dataset.choice = 'skip';
        other.textContent = "Other / I'm not sure";
        predictionButtons.appendChild(other);
        btnEls.push(other);

        function onClick(ev) {
            var choice = ev.currentTarget.dataset.choice;
            btnEls.forEach(function (bb) { bb.classList.add('disabled'); });
            ev.currentTarget.classList.remove('disabled');
            ev.currentTarget.style.outline = '2px solid var(--secondary)';
            socket.emit('prediction_response', {
                session_id: currentDemoSessionId,
                choice: choice
            });
        }

        setTimeout(function () {
            btnEls.forEach(function (bb) {
                bb.classList.remove('disabled');
                bb.addEventListener('click', onClick);
            });
        }, freezeMs);
    }

    function revealPrediction(result) {
        if (!predictionWidget) return;
        var btns = predictionButtons.querySelectorAll('.prediction-btn');
        btns.forEach(function (b) {
            b.classList.add('disabled');
            b.style.outline = '';
            var choice = b.dataset.choice;
            if (choice === String(result.actual_action)) {
                b.classList.add('actual');
            }
            if (result.user_choice !== undefined &&
                choice === String(result.user_choice)) {
                b.classList.add(result.correct ? 'correct' : 'wrong');
            }
        });
        if (predictionReveal) {
            predictionReveal.style.display = '';
            predictionReveal.className = 'prediction-reveal ' +
                (result.correct ? 'correct' : 'wrong');
            predictionReveal.textContent = result.correct
                ? 'Correct! The agent chose action ' + result.actual_action + '.'
                : 'Not quite — the agent chose action ' + result.actual_action + '.';
        }
        // Auto-dismiss so play resumes smoothly.
        setTimeout(hidePredictionWidget, 1200);
    }

    function hidePredictionWidget() {
        awaitingPrediction = false;
        if (predictionWidget) predictionWidget.style.display = 'none';
        if (predictionButtons) predictionButtons.innerHTML = '';
        if (predictionReveal) {
            predictionReveal.style.display = 'none';
            predictionReveal.textContent = '';
            predictionReveal.className = 'prediction-reveal';
        }
        // Restore saliency overlay opacity if Brain-cam is on.
        if (saliencyCanvas && saliencyMode && saliencyMode.value !== 'off' && saliencyOpacity) {
            saliencyCanvas.style.opacity =
                (parseInt(saliencyOpacity.value, 10) / 100).toString();
        }
    }

    function updatePredictionScore(correct, total) {
        var label = correct + ' / ' + total + (total > 0 ? ' correct' : '');
        if (predictionScore) predictionScore.textContent = label;
        if (predictionScoreBadge) predictionScoreBadge.textContent = correct + ' / ' + total;
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
