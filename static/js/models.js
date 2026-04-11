/* ============================================================
   ARENA OF RL - Models Page JS
   ============================================================ */

(function () {
    'use strict';

    var tableBody     = document.getElementById('modelsTableBody');
    var emptyState    = document.getElementById('modelsEmptyState');
    var tableWrapper  = document.getElementById('modelsTableWrapper');
    var countBadge    = document.getElementById('modelCountBadge');
    var btnCompare    = document.getElementById('btnCompare');
    var btnRefresh    = document.getElementById('btnRefreshModels');
    var btnCardView   = document.getElementById('btnCardView');
    var btnTableView  = document.getElementById('btnTableView');
    var compareCount  = document.getElementById('compareCount');
    var compSection   = document.getElementById('comparisonSection');
    var compContent   = document.getElementById('comparisonContent');
    var btnCloseCmp   = document.getElementById('btnCloseComparison');
    var selectAll     = document.getElementById('selectAllModels');
    var cardContainer = document.getElementById('modelsCardContainer');

    var modelsData   = [];
    var selectedIds  = [];
    var viewMode     = 'table';

    // ---- Fetch & Render ----
    function loadModels() {
        fetch('/api/models')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                modelsData = data.models || [];
                renderModels();
            })
            .catch(function () {
                modelsData = [];
                renderModels();
                window.showToast('Failed to load models.', 'error');
            });
    }

    function renderModels() {
        selectedIds = [];
        updateCompareButton();

        if (modelsData.length === 0) {
            tableWrapper.classList.add('hidden');
            if (cardContainer) cardContainer.classList.add('hidden');
            emptyState.classList.remove('hidden');
            countBadge.textContent = '0 models';
            return;
        }

        emptyState.classList.add('hidden');
        countBadge.textContent = modelsData.length + ' model' + (modelsData.length !== 1 ? 's' : '');

        if (viewMode === 'table') {
            renderTableView();
        } else {
            renderCardView();
        }
    }

    function renderTableView() {
        tableWrapper.classList.remove('hidden');
        if (cardContainer) cardContainer.classList.add('hidden');

        tableBody.innerHTML = '';
        modelsData.forEach(function (model, idx) {
            var tr = document.createElement('tr');
            // BUG-008 FIX: use total_timesteps and created_at
            tr.innerHTML =
                '<td><input type="checkbox" class="model-check" data-idx="' + idx + '"></td>' +
                '<td><strong>' + escHtml(model.name || 'Unnamed') + '</strong></td>' +
                '<td>' + escHtml(model.scenario || '--') + '</td>' +
                '<td class="mono">' + window.formatNumber(model.total_timesteps || 0, 0) + '</td>' +
                '<td class="mono" style="color:var(--secondary);">' + window.formatNumber(model.mean_reward, 2) + '</td>' +
                '<td class="text-muted">' + formatDate(model.created_at) + '</td>' +
                '<td>' +
                    '<div class="btn-group">' +
                        '<a href="/play?model=' + encodeURIComponent(model.id || '') + '" class="btn btn-sm btn-secondary"><i class="fas fa-gamepad"></i> Demo</a>' +
                        '<a href="/evaluate?model=' + encodeURIComponent(model.id || '') + '" class="btn btn-sm btn-outline"><i class="fas fa-chart-bar"></i> Eval</a>' +
                        '<button class="btn btn-sm btn-danger btn-delete" data-id="' + escAttr(model.id || '') + '" data-name="' + escAttr(model.name || '') + '"><i class="fas fa-trash"></i></button>' +
                    '</div>' +
                '</td>';
            tableBody.appendChild(tr);
        });

        wireListeners();
    }

    function renderCardView() {
        tableWrapper.classList.add('hidden');
        if (!cardContainer) return;
        cardContainer.classList.remove('hidden');

        cardContainer.innerHTML = '';
        modelsData.forEach(function (model, idx) {
            var rewardColor = (model.mean_reward || 0) >= 0 ? 'var(--success)' : 'var(--danger)';
            var card = document.createElement('div');
            card.className = 'model-card animate-slideIn';
            // BUG-008 FIX: use total_timesteps and created_at
            card.innerHTML =
                '<div class="model-card-header">' +
                    '<h4>' + escHtml(model.name || 'Unnamed') + '</h4>' +
                    '<span class="cfg-tag">' + escHtml(model.scenario || 'unknown') + '</span>' +
                '</div>' +
                '<div class="model-card-stats">' +
                    '<div class="model-card-stat">' +
                        '<span class="label">Reward</span>' +
                        '<span class="value" style="color:' + rewardColor + ';">' + window.formatNumber(model.mean_reward, 2) + '</span>' +
                    '</div>' +
                    '<div class="model-card-stat">' +
                        '<span class="label">Timesteps</span>' +
                        '<span class="value">' + window.formatNumber(model.total_timesteps || 0, 0) + '</span>' +
                    '</div>' +
                    '<div class="model-card-stat">' +
                        '<span class="label">Created</span>' +
                        '<span class="value">' + formatDate(model.created_at) + '</span>' +
                    '</div>' +
                '</div>' +
                '<div class="model-card-actions">' +
                    '<a href="/play?model=' + encodeURIComponent(model.id || '') + '" class="btn btn-sm btn-primary"><i class="fas fa-gamepad"></i> Play</a>' +
                    '<a href="/evaluate?model=' + encodeURIComponent(model.id || '') + '" class="btn btn-sm btn-outline"><i class="fas fa-chart-bar"></i> Eval</a>' +
                    '<button class="btn btn-sm btn-danger btn-delete" data-id="' + escAttr(model.id || '') + '" data-name="' + escAttr(model.name || '') + '"><i class="fas fa-trash"></i></button>' +
                '</div>';
            cardContainer.appendChild(card);
        });

        wireListeners();
    }

    function wireListeners() {
        document.querySelectorAll('.model-check').forEach(function (cb) {
            cb.addEventListener('change', onCheckboxChange);
        });
        document.querySelectorAll('.btn-delete').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var id = this.dataset.id;
                var name = this.dataset.name;
                if (confirm('Delete model "' + name + '"? This cannot be undone.')) {
                    deleteModel(id);
                }
            });
        });
    }

    // ---- View Toggle ----
    if (btnCardView) {
        btnCardView.addEventListener('click', function () {
            viewMode = 'card';
            btnCardView.classList.add('active');
            if (btnTableView) btnTableView.classList.remove('active');
            renderModels();
        });
    }
    if (btnTableView) {
        btnTableView.addEventListener('click', function () {
            viewMode = 'table';
            btnTableView.classList.add('active');
            if (btnCardView) btnCardView.classList.remove('active');
            renderModels();
        });
    }

    // ---- Checkbox Selection ----
    function onCheckboxChange() {
        selectedIds = [];
        document.querySelectorAll('.model-check:checked').forEach(function (cb) {
            selectedIds.push(parseInt(cb.dataset.idx));
        });
        updateCompareButton();
    }

    function updateCompareButton() {
        compareCount.textContent = selectedIds.length;
        btnCompare.disabled = selectedIds.length !== 2;
    }

    if (selectAll) {
        selectAll.addEventListener('change', function () {
            var checked = this.checked;
            document.querySelectorAll('.model-check').forEach(function (cb) {
                cb.checked = checked;
            });
            onCheckboxChange();
        });
    }

    // ---- Delete Model ----
    function deleteModel(id) {
        fetch('/api/models/' + encodeURIComponent(id), { method: 'DELETE' })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.success) {
                    window.showToast('Model deleted.', 'success');
                    loadModels();
                } else {
                    window.showToast('Failed to delete model: ' + (data.error || data.message || ''), 'error');
                }
            })
            .catch(function () {
                window.showToast('Network error deleting model.', 'error');
            });
    }

    // ---- Model Comparison ----
    btnCompare.addEventListener('click', function () {
        if (selectedIds.length !== 2) return;
        var a = modelsData[selectedIds[0]];
        var b = modelsData[selectedIds[1]];
        showComparison(a, b);
    });

    btnCloseCmp.addEventListener('click', function () {
        compSection.classList.add('hidden');
    });

    function showComparison(a, b) {
        compContent.innerHTML =
            buildComparisonCard(a) + buildComparisonCard(b);
        compSection.classList.remove('hidden');
        compSection.scrollIntoView({ behavior: 'smooth' });
    }

    function buildComparisonCard(model) {
        // BUG-008 FIX: use total_timesteps and created_at
        return '<div class="comparison-card">' +
            '<h4>' + escHtml(model.name || 'Unnamed') + '</h4>' +
            compRow('Scenario',   model.scenario || '--') +
            compRow('Timesteps',  window.formatNumber(model.total_timesteps || 0, 0)) +
            compRow('Mean Reward', window.formatNumber(model.mean_reward, 2)) +
            compRow('Created',    formatDate(model.created_at)) +
            compRow('Algorithm',  model.algorithm || 'PPO') +
            '</div>';
    }

    function compRow(label, value) {
        return '<div class="comparison-stat">' +
            '<span class="label">' + label + '</span>' +
            '<span class="value">' + value + '</span>' +
            '</div>';
    }

    // ---- Refresh ----
    btnRefresh.addEventListener('click', function () {
        loadModels();
        window.showToast('Models refreshed.', 'info');
    });

    // ---- Helpers ----
    function escHtml(str) {
        var d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }
    function escAttr(str) {
        return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
    function formatDate(dateStr) {
        if (!dateStr || dateStr === '--') return '--';
        try {
            var d = new Date(dateStr);
            return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        } catch(e) {
            return dateStr;
        }
    }

    // ---- Init ----
    loadModels();

})();
