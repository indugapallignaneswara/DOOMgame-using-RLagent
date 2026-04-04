/* ============================================================
   DOOM RL – Models Page JS
   ============================================================ */

(function () {
    'use strict';

    var tableBody     = document.getElementById('modelsTableBody');
    var emptyState    = document.getElementById('modelsEmptyState');
    var tableWrapper  = document.getElementById('modelsTableWrapper');
    var countBadge    = document.getElementById('modelCountBadge');
    var btnCompare    = document.getElementById('btnCompare');
    var btnRefresh    = document.getElementById('btnRefreshModels');
    var compareCount  = document.getElementById('compareCount');
    var compSection   = document.getElementById('comparisonSection');
    var compContent   = document.getElementById('comparisonContent');
    var btnCloseCmp   = document.getElementById('btnCloseComparison');
    var selectAll     = document.getElementById('selectAllModels');

    var modelsData   = [];
    var selectedIds  = [];

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
            emptyState.classList.remove('hidden');
            countBadge.textContent = '0 models';
            return;
        }

        tableWrapper.classList.remove('hidden');
        emptyState.classList.add('hidden');
        countBadge.textContent = modelsData.length + ' model' + (modelsData.length !== 1 ? 's' : '');

        tableBody.innerHTML = '';
        modelsData.forEach(function (model, idx) {
            var tr = document.createElement('tr');
            tr.innerHTML =
                '<td><input type="checkbox" class="model-check" data-idx="' + idx + '"></td>' +
                '<td><strong>' + escHtml(model.name || 'Unnamed') + '</strong></td>' +
                '<td>' + escHtml(model.scenario || '--') + '</td>' +
                '<td class="mono">' + window.formatNumber(model.timesteps || 0, 0) + '</td>' +
                '<td class="mono" style="color:var(--secondary);">' + window.formatNumber(model.mean_reward, 2) + '</td>' +
                '<td class="text-muted">' + escHtml(model.created || '--') + '</td>' +
                '<td>' +
                    '<div class="btn-group">' +
                        '<a href="/play?model=' + encodeURIComponent(model.name || model.path || '') + '" class="btn btn-sm btn-secondary"><i class="fas fa-gamepad"></i> Demo</a>' +
                        '<a href="/evaluate?model=' + encodeURIComponent(model.name || model.path || '') + '" class="btn btn-sm btn-outline"><i class="fas fa-chart-bar"></i> Eval</a>' +
                        '<button class="btn btn-sm btn-danger btn-delete" data-name="' + escAttr(model.name || '') + '"><i class="fas fa-trash"></i></button>' +
                    '</div>' +
                '</td>';
            tableBody.appendChild(tr);
        });

        // Wire checkbox listeners
        document.querySelectorAll('.model-check').forEach(function (cb) {
            cb.addEventListener('change', onCheckboxChange);
        });

        // Wire delete buttons
        document.querySelectorAll('.btn-delete').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var name = this.dataset.name;
                if (confirm('Delete model "' + name + '"? This cannot be undone.')) {
                    deleteModel(name);
                }
            });
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
    function deleteModel(name) {
        fetch('/api/models/' + encodeURIComponent(name), { method: 'DELETE' })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.success) {
                    window.showToast('Model "' + name + '" deleted.', 'success');
                    loadModels();
                } else {
                    window.showToast('Failed to delete model: ' + (data.message || ''), 'error');
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
        return '<div class="comparison-card">' +
            '<h4>' + escHtml(model.name || 'Unnamed') + '</h4>' +
            compRow('Scenario',   model.scenario || '--') +
            compRow('Timesteps',  window.formatNumber(model.timesteps || 0, 0)) +
            compRow('Mean Reward', window.formatNumber(model.mean_reward, 2)) +
            compRow('Created',    model.created || '--') +
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

    // ---- Init ----
    loadModels();

})();
