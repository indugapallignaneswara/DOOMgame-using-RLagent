/* ============================================================
   ARENA OF RL — Mind Map: See What Your AI Thinks
   ============================================================ */

(function () {
    'use strict';

    var socket = window.socket;
    var modelSelect  = document.getElementById('mmModelSelect');
    var btnStart     = document.getElementById('btnStartMM');
    var btnStop      = document.getElementById('btnStopMM');
    var canvas       = document.getElementById('mmCanvas');
    var ctx          = canvas ? canvas.getContext('2d') : null;
    var overlay      = document.getElementById('mmOverlay');
    var gradCamCanvas= document.getElementById('mmGradCamCanvas');
    var gradCamCtx   = gradCamCanvas ? gradCamCanvas.getContext('2d') : null;
    var gradCamToggle= document.getElementById('mmGradCamToggle');

    // Stats
    var elAction  = document.getElementById('mmAction');
    var elReward  = document.getElementById('mmReward');
    var elStep    = document.getElementById('mmStep');
    var elEpisode = document.getElementById('mmEpisode');

    // Decision panel
    var elActionBars  = document.getElementById('mmActionBars');
    var elValue       = document.getElementById('mmValue');
    var elEntropy     = document.getElementById('mmEntropy');
    var elConfidence  = document.getElementById('mmConfidence');
    var elAlternatives= document.getElementById('mmAlternatives');
    var valueSpark    = document.getElementById('mmValueSpark');
    var valueSparkCtx = valueSpark ? valueSpark.getContext('2d') : null;

    // Channel grid
    var channelGrid = document.getElementById('mmChannelGrid');

    var sessionId = null;
    var isRunning = false;
    var frameImage = new Image();
    var gradCamImage = new Image();
    var valueHistory = [];
    var buttons = [];

    // Timeline data buffers
    var timelineData = [];     // [{step, channel_means, action, value, reward, kl_div, is_critical}]
    var criticalMoments = [];
    var ACTION_COLORS = ['#ff4400','#3388ff','#44cc44','#ffaa00','#cc44ff','#ff44aa','#00f0ff','#ff8844'];

    // Timeline canvases
    var timelineCanvas = document.getElementById('mmTimelineCanvas');
    var timelineCtx = timelineCanvas ? timelineCanvas.getContext('2d') : null;
    var actionTimeline = document.getElementById('mmActionTimeline');
    var actionTimelineCtx = actionTimeline ? actionTimeline.getContext('2d') : null;
    var valueTimeline = document.getElementById('mmValueTimeline');
    var valueTimelineCtx = valueTimeline ? valueTimeline.getContext('2d') : null;
    var criticalList = document.getElementById('mmCriticalMoments');

    // ---- Tab Switching ----
    var tabs = document.querySelectorAll('.mm-tab');
    var panels = document.querySelectorAll('.mm-panel');
    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            tabs.forEach(function (t) { t.classList.remove('active'); });
            panels.forEach(function (p) { p.classList.add('hidden'); });
            tab.classList.add('active');
            var target = document.getElementById('panel' + capitalize(tab.dataset.tab));
            if (target) target.classList.remove('hidden');
        });
    });

    function capitalize(s) {
        return s.charAt(0).toUpperCase() + s.slice(1);
    }

    // ---- Load Models ----
    fetch('/api/models')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            modelSelect.innerHTML = '<option value="">-- Select Model --</option>';
            if (data.models && data.models.length > 0) {
                data.models.forEach(function (m) {
                    var opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.name + (m.scenario ? ' (' + m.scenario + ')' : '');
                    modelSelect.appendChild(opt);
                });
            }
        });

    // ---- Start ----
    btnStart.addEventListener('click', function () {
        var modelId = modelSelect.value;
        if (!modelId) { window.showToast('Select a model first.', 'warning'); return; }
        btnStart.disabled = true;
        btnStart.innerHTML = '<span class="spinner-inline"></span> Loading...';
        socket.emit('start_mindmap', { model_id: modelId });
    });

    // ---- Stop ----
    btnStop.addEventListener('click', function () {
        if (sessionId) socket.emit('stop_mindmap', { session_id: sessionId });
    });

    // ---- Status ----
    socket.on('mindmap_status', function (data) {
        if (data.session_id) sessionId = data.session_id;
        if (data.status === 'running' || data.status === 'started') {
            isRunning = true;
            btnStart.disabled = true;
            btnStart.innerHTML = '<i class="fas fa-eye"></i> Running';
            btnStop.disabled = false;
            if (overlay) overlay.style.display = 'none';
            if (data.buttons) buttons = data.buttons;
            buildActionBars(buttons);
        } else if (data.status === 'stopped' || data.status === 'error') {
            isRunning = false;
            btnStart.disabled = false;
            btnStart.innerHTML = '<i class="fas fa-eye"></i> Launch';
            btnStop.disabled = true;
            if (data.status === 'error') {
                window.showToast('Mind Map error: ' + (data.error || 'Unknown'), 'error');
            }
        }
    });

    // ---- Main Frame Handler ----
    socket.on('mindmap_frame', function (data) {
        if (!isRunning) return;

        // Game frame
        if (data.frame && ctx) {
            frameImage.onload = function () {
                ctx.drawImage(frameImage, 0, 0, canvas.width, canvas.height);
            };
            frameImage.src = 'data:image/jpeg;base64,' + data.frame;
        }

        // Grad-CAM overlay
        if (data.grad_cam && gradCamCtx && gradCamToggle.checked) {
            gradCamCanvas.style.display = 'block';
            gradCamImage.onload = function () {
                gradCamCtx.clearRect(0, 0, gradCamCanvas.width, gradCamCanvas.height);
                gradCamCtx.globalAlpha = 0.45;
                gradCamCtx.drawImage(gradCamImage, 0, 0, gradCamCanvas.width, gradCamCanvas.height);
                gradCamCtx.globalAlpha = 1.0;
            };
            gradCamImage.src = 'data:image/png;base64,' + data.grad_cam;
        } else if (gradCamCtx) {
            if (!gradCamToggle.checked) {
                gradCamCtx.clearRect(0, 0, gradCamCanvas.width, gradCamCanvas.height);
                gradCamCanvas.style.display = 'none';
            }
        }

        // Stats bar
        elAction.textContent = data.action_name || '--';
        elReward.textContent = data.total_reward;
        elStep.textContent = data.step;
        elEpisode.textContent = data.episode;

        // ─── Decision Panel ───
        updateActionBars(data.action_probs, data.action, data.buttons || buttons);
        elValue.textContent = data.value;
        elEntropy.textContent = data.entropy;

        // Confidence (inverse entropy)
        var maxEntropy = Math.log(Math.max((data.action_probs || []).length, 2));
        var confidence = 1.0 - (data.entropy / maxEntropy);
        confidence = Math.max(0, Math.min(1, confidence));
        if (elConfidence) elConfidence.style.width = (confidence * 100) + '%';

        // Value sparkline
        valueHistory.push(data.value);
        if (valueHistory.length > 50) valueHistory.shift();
        drawSparkline(valueSparkCtx, valueSpark, valueHistory);

        // Alternatives
        updateAlternatives(data.action_probs, data.action, data.buttons || buttons);

        // ─── X-Ray Panel ───
        if (data.activations) {
            drawHeatmap('mmConv1Map', data.activations.conv1);
            drawHeatmap('mmConv2Map', data.activations.conv2);
            drawHeatmap('mmConv3Map', data.activations.conv3);
            drawImageToCanvas('mmAttConv1', data.activations.conv1);
            drawImageToCanvas('mmAttConv2', data.activations.conv2);
            drawImageToCanvas('mmAttConv3', data.activations.conv3);

            // Layer stats
            if (data.channel_means) {
                updateLayerStats('mmConv1', data.channel_means.conv1, 32);
                updateLayerStats('mmConv2', data.channel_means.conv2, 64);
                updateLayerStats('mmConv3', data.channel_means.conv3, 64);
            }

            // Top channels with descriptive labels
            if (data.activations.top_channels && channelGrid) {
                channelGrid.innerHTML = '';
                data.activations.top_channels.forEach(function (ch, i) {
                    var label = describeChannel(ch.idx, ch.mean, i);
                    var div = document.createElement('div');
                    div.className = 'mm-channel-cell';
                    div.innerHTML = '<img src="data:image/png;base64,' + ch.data + '" alt="ch' + ch.idx + '">' +
                        '<span class="mm-ch-label">' + label + '</span>' +
                        '<span class="mm-ch-val">' + ch.mean.toFixed(2) + '</span>';
                    channelGrid.appendChild(div);
                });
            }
        }

        // ─── AI Reasoning Summary ───
        updateReasoning(data);
        updateDecisionSummary(data);

        // ─── Attention Panel ───
        if (data.frame) drawImageToCanvas('mmAttRaw', data.frame, 'jpeg');
        if (data.grad_cam) drawImageToCanvas('mmAttCam', data.grad_cam, 'png');

        // Input preview (small)
        if (data.frame) drawImageToCanvas('mmInputPreview', data.frame, 'jpeg');

        // ─── Attention Trail on Game Canvas ───
        if (data.attention_trail && ctx) {
            var trail = data.attention_trail;
            if (trail.length > 1) {
                ctx.save();
                for (var t = 1; t < trail.length; t++) {
                    var alpha = t / trail.length;
                    var x1 = trail[t-1].x * canvas.width;
                    var y1 = trail[t-1].y * canvas.height;
                    var x2 = trail[t].x * canvas.width;
                    var y2 = trail[t].y * canvas.height;
                    ctx.beginPath();
                    ctx.strokeStyle = 'rgba(0, 240, 255, ' + (alpha * 0.8) + ')';
                    ctx.lineWidth = 1 + alpha * 2;
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }
                // Current attention point — pulsing dot
                var last = trail[trail.length - 1];
                var px = last.x * canvas.width;
                var py = last.y * canvas.height;
                ctx.beginPath();
                ctx.arc(px, py, 6, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(0, 240, 255, 0.9)';
                ctx.fill();
                ctx.beginPath();
                ctx.arc(px, py, 10, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
                ctx.lineWidth = 2;
                ctx.stroke();
                ctx.restore();
            }
        }

        // ─── Critical Moment Flash ───
        if (data.is_critical && ctx) {
            ctx.save();
            ctx.fillStyle = 'rgba(255, 68, 0, 0.15)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.restore();
        }

        // ─── Timeline Data Accumulation ───
        timelineData.push({
            step: data.step,
            channel_means: data.channel_means || {},
            action: data.action,
            value: data.value,
            reward: data.reward,
            kl_div: data.kl_div || 0,
            is_critical: data.is_critical || false,
        });
        if (timelineData.length > 200) timelineData.shift();

        if (data.is_critical) {
            criticalMoments.push({
                step: data.step, episode: data.episode,
                kl: data.kl_div, action: data.action_name,
                value: data.value,
            });
            if (criticalMoments.length > 20) criticalMoments.shift();
        }

        // Render timelines every 3 frames for performance
        if (data.step % 3 === 0) {
            renderTimeline();
            renderActionTimeline();
            renderValueTimeline();
            renderCriticalList();
        }
    });

    // ---- Episode End ----
    socket.on('mindmap_episode_end', function (data) {
        valueHistory = [];
    });

    // ============================================================
    //  RENDERING HELPERS
    // ============================================================

    function buildActionBars(btns) {
        if (!elActionBars) return;
        elActionBars.innerHTML = '';
        (btns || []).forEach(function (name, i) {
            var row = document.createElement('div');
            row.className = 'mm-action-row';
            row.innerHTML =
                '<span class="mm-action-label">' + name + '</span>' +
                '<div class="mm-action-bar-bg"><div class="mm-action-bar-fill" id="mmBar' + i + '"></div></div>' +
                '<span class="mm-action-pct" id="mmPct' + i + '">0%</span>';
            elActionBars.appendChild(row);
        });
    }

    function updateActionBars(probs, chosenAction, btns) {
        if (!probs || !elActionBars) return;
        probs.forEach(function (p, i) {
            var bar = document.getElementById('mmBar' + i);
            var pct = document.getElementById('mmPct' + i);
            if (bar) {
                bar.style.width = (p * 100) + '%';
                bar.className = 'mm-action-bar-fill' + (i === chosenAction ? ' chosen' : '');
            }
            if (pct) pct.textContent = (p * 100).toFixed(1) + '%';
        });
    }

    function updateAlternatives(probs, chosen, btns) {
        if (!probs || !elAlternatives) return;
        var indexed = probs.map(function (p, i) { return { i: i, p: p }; });
        indexed.sort(function (a, b) { return b.p - a.p; });
        elAlternatives.innerHTML = '';
        var count = 0;
        for (var k = 0; k < indexed.length && count < 3; k++) {
            var item = indexed[k];
            if (item.i === chosen) continue;
            var name = (btns && btns[item.i]) ? btns[item.i] : 'Act ' + item.i;
            var card = document.createElement('div');
            card.className = 'mm-alt-card';
            card.innerHTML = '<span class="mm-alt-name">' + name + '</span>' +
                '<span class="mm-alt-prob">' + (item.p * 100).toFixed(1) + '%</span>';
            elAlternatives.appendChild(card);
            count++;
        }
    }

    function drawSparkline(sparkCtx, sparkCanvas, data) {
        if (!sparkCtx || !sparkCanvas || data.length < 2) return;
        var w = sparkCanvas.width, h = sparkCanvas.height;
        sparkCtx.clearRect(0, 0, w, h);
        var min = Math.min.apply(null, data);
        var max = Math.max.apply(null, data);
        var range = max - min || 1;
        sparkCtx.beginPath();
        sparkCtx.strokeStyle = '#00f0ff';
        sparkCtx.lineWidth = 1.5;
        for (var i = 0; i < data.length; i++) {
            var x = (i / (data.length - 1)) * w;
            var y = h - ((data[i] - min) / range) * (h - 4) - 2;
            if (i === 0) sparkCtx.moveTo(x, y);
            else sparkCtx.lineTo(x, y);
        }
        sparkCtx.stroke();
    }

    function drawHeatmap(canvasId, imgB64) {
        if (!imgB64) return;
        var c = document.getElementById(canvasId);
        if (!c) return;
        var hctx = c.getContext('2d');
        var img = new Image();
        img.onload = function () {
            hctx.clearRect(0, 0, c.width, c.height);
            hctx.drawImage(img, 0, 0, c.width, c.height);
        };
        img.src = 'data:image/png;base64,' + imgB64;
    }

    function drawImageToCanvas(canvasId, imgB64, format) {
        if (!imgB64) return;
        var c = document.getElementById(canvasId);
        if (!c) return;
        var ictx = c.getContext('2d');
        var img = new Image();
        img.onload = function () {
            ictx.clearRect(0, 0, c.width, c.height);
            ictx.drawImage(img, 0, 0, c.width, c.height);
        };
        img.src = 'data:image/' + (format || 'png') + ';base64,' + imgB64;
    }

    // ============================================================
    //  EXPLANATORY HELPERS
    // ============================================================

    function updateLayerStats(prefix, means, total) {
        if (!means) return;
        var activeEl = document.getElementById(prefix + 'Active');
        var peakEl = document.getElementById(prefix + 'Peak');
        if (!activeEl || !peakEl) return;
        var active = means.filter(function(v) { return v > 0.1; }).length;
        var peak = Math.max.apply(null, means);
        // Extrapolate: if top 8 have X active, estimate for full layer
        var estActive = Math.round(active * total / 8);
        activeEl.textContent = estActive;
        peakEl.textContent = peak.toFixed(2);
    }

    function describeChannel(idx, mean, rank) {
        // Heuristic labels based on activation strength and rank
        var descriptions = [
            'Primary detector', 'Strong signal', 'Active region',
            'Motion tracker', 'Edge response', 'Texture pattern',
            'Spatial feature', 'Background scan'
        ];
        return 'Filter #' + idx;
    }

    function updateReasoning(data) {
        var el = document.getElementById('mmReasoningText');
        if (!el) return;

        var action = data.action_name || 'unknown';
        var prob = data.action_probs ? (data.action_probs[data.action] * 100).toFixed(0) : '?';
        var confidence = '';
        var maxEntropy = Math.log(Math.max((data.action_probs || []).length, 2));
        var confPct = 1.0 - ((data.entropy || 0) / maxEntropy);
        confPct = Math.max(0, Math.min(1, confPct));

        if (confPct > 0.8) confidence = 'Very confident';
        else if (confPct > 0.6) confidence = 'Fairly sure';
        else if (confPct > 0.4) confidence = 'Somewhat uncertain';
        else confidence = 'Very uncertain';

        var valueSentiment = '';
        var v = data.value || 0;
        if (v > 5) valueSentiment = 'Feels very optimistic about this position.';
        else if (v > 1) valueSentiment = 'Thinks this situation is favorable.';
        else if (v > -1) valueSentiment = 'Neutral about current position.';
        else if (v > -5) valueSentiment = 'Slightly worried about this situation.';
        else valueSentiment = 'Thinks this is a dangerous position.';

        var attention = '';
        if (data.attn_cx !== undefined) {
            var cx = data.attn_cx, cy = data.attn_cy;
            if (cx < 0.35) attention = 'Looking left.';
            else if (cx > 0.65) attention = 'Looking right.';
            else attention = 'Focused on center.';
            if (cy < 0.35) attention += ' Eyes up.';
            else if (cy > 0.65) attention += ' Eyes down.';
        }

        var critical = data.is_critical ? ' <span style="color:var(--primary);">POLICY SHIFT — changing strategy!</span>' : '';

        el.innerHTML =
            '<strong style="color:var(--secondary);">' + action + '</strong> (' + prob + '% probability) — ' +
            confidence + '. ' + valueSentiment + ' ' + attention + critical;
    }

    function updateDecisionSummary(data) {
        var el = document.getElementById('mmDecisionText');
        if (!el || !data.action_probs) return;

        var probs = data.action_probs;
        var btns = data.buttons || buttons;
        var chosen = data.action;
        var chosenName = btns[chosen] || 'Action ' + chosen;
        var chosenProb = (probs[chosen] * 100).toFixed(1);

        // Find runner-up
        var sorted = probs.map(function(p, i) { return {i:i, p:p}; }).sort(function(a,b) { return b.p - a.p; });
        var runnerUp = sorted[0].i === chosen ? sorted[1] : sorted[0];
        var runnerName = btns[runnerUp.i] || 'Action ' + runnerUp.i;
        var runnerProb = (runnerUp.p * 100).toFixed(1);
        var gap = (probs[chosen] - runnerUp.p) * 100;

        var verdict = '';
        if (gap > 40) verdict = 'Clear decision — no contest.';
        else if (gap > 20) verdict = 'Confident choice with some alternatives.';
        else if (gap > 5) verdict = 'Close call — almost went with ' + runnerName + '.';
        else verdict = '<span style="color:var(--warning);">Coin flip!</span> Nearly chose ' + runnerName + ' instead.';

        el.innerHTML =
            'Chose <strong style="color:var(--secondary);">' + chosenName + '</strong> at ' + chosenProb +
            '%. Runner-up: ' + runnerName + ' at ' + runnerProb + '%. ' + verdict;
    }

    // ============================================================
    //  TIMELINE HEATMAP RENDERING
    // ============================================================

    function renderTimeline() {
        if (!timelineCtx || timelineData.length < 2) return;
        var c = timelineCanvas;
        var w = c.width, h = c.height;
        timelineCtx.clearRect(0, 0, w, h);

        // Collect all channel names and build rows
        var rows = [];
        var layerNames = ['conv1', 'conv2', 'conv3'];
        for (var li = 0; li < layerNames.length; li++) {
            var ln = layerNames[li];
            for (var ci = 0; ci < 8; ci++) {
                rows.push({ layer: ln, ch: ci, label: ln + '#' + ci });
            }
        }

        var numRows = rows.length; // 24
        var numCols = timelineData.length;
        var cellW = Math.max(2, w / numCols);
        var cellH = h / numRows;

        for (var col = 0; col < numCols; col++) {
            var d = timelineData[col];
            for (var row = 0; row < numRows; row++) {
                var r = rows[row];
                var val = 0;
                if (d.channel_means && d.channel_means[r.layer] && d.channel_means[r.layer][r.ch] !== undefined) {
                    val = d.channel_means[r.layer][r.ch];
                }
                // Normalize to 0-1 (rough, using 2.0 as typical max)
                val = Math.min(val / 2.0, 1.0);
                var color = heatColor(val);
                timelineCtx.fillStyle = color;
                timelineCtx.fillRect(col * cellW, row * cellH, cellW + 0.5, cellH + 0.5);
            }

            // Mark critical moments with a red vertical line
            if (d.is_critical) {
                timelineCtx.fillStyle = 'rgba(255, 68, 0, 0.6)';
                timelineCtx.fillRect(col * cellW, 0, 2, h);
            }
        }

        // Current position indicator
        timelineCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        timelineCtx.fillRect((numCols - 1) * cellW, 0, 2, h);

        // Layer labels on left
        timelineCtx.font = '9px monospace';
        timelineCtx.fillStyle = 'rgba(255,255,255,0.5)';
        for (var li = 0; li < layerNames.length; li++) {
            var y = li * 8 * cellH + 4 * cellH;
            timelineCtx.fillText(layerNames[li], 2, y + 3);
        }
    }

    function renderActionTimeline() {
        if (!actionTimelineCtx || timelineData.length < 2) return;
        var c = actionTimeline;
        var w = c.width, h = c.height;
        actionTimelineCtx.clearRect(0, 0, w, h);
        var cellW = Math.max(2, w / timelineData.length);

        for (var i = 0; i < timelineData.length; i++) {
            var action = timelineData[i].action || 0;
            actionTimelineCtx.fillStyle = ACTION_COLORS[action % ACTION_COLORS.length];
            actionTimelineCtx.fillRect(i * cellW, 0, cellW + 0.5, h);
        }
    }

    function renderValueTimeline() {
        if (!valueTimelineCtx || timelineData.length < 2) return;
        var c = valueTimeline;
        var w = c.width, h = c.height;
        valueTimelineCtx.clearRect(0, 0, w, h);

        // Draw value line
        var values = timelineData.map(function (d) { return d.value; });
        var min = Math.min.apply(null, values);
        var max = Math.max.apply(null, values);
        var range = max - min || 1;

        valueTimelineCtx.beginPath();
        valueTimelineCtx.strokeStyle = '#00f0ff';
        valueTimelineCtx.lineWidth = 1.5;
        for (var i = 0; i < values.length; i++) {
            var x = (i / (values.length - 1)) * w;
            var y = h - ((values[i] - min) / range) * (h - 8) - 4;
            if (i === 0) valueTimelineCtx.moveTo(x, y);
            else valueTimelineCtx.lineTo(x, y);
        }
        valueTimelineCtx.stroke();

        // Draw reward dots
        for (var i = 0; i < timelineData.length; i++) {
            var r = timelineData[i].reward;
            if (Math.abs(r) > 0.5) {
                var x = (i / (values.length - 1)) * w;
                valueTimelineCtx.beginPath();
                valueTimelineCtx.arc(x, r > 0 ? 8 : h - 8, 3, 0, Math.PI * 2);
                valueTimelineCtx.fillStyle = r > 0 ? '#00ff9d' : '#ff2244';
                valueTimelineCtx.fill();
            }
        }

        // Labels
        valueTimelineCtx.font = '9px monospace';
        valueTimelineCtx.fillStyle = 'rgba(255,255,255,0.4)';
        valueTimelineCtx.fillText('V(s): ' + values[values.length-1].toFixed(2), 4, 12);
    }

    function renderCriticalList() {
        if (!criticalList) return;
        if (criticalMoments.length === 0) {
            criticalList.innerHTML = '<span class="text-muted" style="font-size:0.75rem;">No critical moments yet...</span>';
            return;
        }
        criticalList.innerHTML = '';
        var recent = criticalMoments.slice(-8).reverse();
        for (var i = 0; i < recent.length; i++) {
            var m = recent[i];
            var div = document.createElement('div');
            div.className = 'mm-critical-item';
            div.innerHTML =
                '<span class="mm-critical-step">Step ' + m.step + '</span>' +
                '<span class="mm-critical-action">' + m.action + '</span>' +
                '<span class="mm-critical-kl">KL: ' + m.kl.toFixed(3) + '</span>';
            criticalList.appendChild(div);
        }
    }

    function heatColor(val) {
        // black → deep blue → cyan → white
        val = Math.max(0, Math.min(1, val));
        if (val < 0.33) {
            var t = val / 0.33;
            return 'rgb(' + Math.round(t*10) + ',' + Math.round(t*20) + ',' + Math.round(30 + t*80) + ')';
        } else if (val < 0.66) {
            var t = (val - 0.33) / 0.33;
            return 'rgb(' + Math.round(10 + t*0) + ',' + Math.round(20 + t*220) + ',' + Math.round(110 + t*145) + ')';
        } else {
            var t = (val - 0.66) / 0.34;
            return 'rgb(' + Math.round(10 + t*245) + ',' + Math.round(240 + t*15) + ',' + Math.round(255) + ')';
        }
    }

})();
