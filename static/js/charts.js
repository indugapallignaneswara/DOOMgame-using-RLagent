/* ============================================================
   DOOM RL – Chart.js Wrappers
   ============================================================ */

/* Dark theme defaults */
Chart.defaults.color = '#a0a0b0';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.animation.duration = 400;

/**
 * Create a line chart for episode reward tracking.
 * Returns the Chart instance.
 */
function createRewardChart(canvasId) {
    var ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    ctx = ctx.getContext('2d');

    var gradient = ctx.createLinearGradient(0, 0, 0, 280);
    gradient.addColorStop(0, 'rgba(0, 240, 255, 0.25)');
    gradient.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Episode Reward',
                data: [],
                borderColor: '#00f0ff',
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 4,
                pointHoverBackgroundColor: '#00f0ff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { maxTicksLimit: 10, color: '#606070' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#606070' }
                }
            },
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(20, 20, 35, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#00f0ff',
                    borderColor: 'rgba(0, 240, 255, 0.3)',
                    borderWidth: 1,
                    padding: 10
                }
            }
        }
    });
}

/**
 * Create a line chart for loss tracking.
 * Returns the Chart instance.
 */
function createLossChart(canvasId) {
    var ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    ctx = ctx.getContext('2d');

    var gradient = ctx.createLinearGradient(0, 0, 0, 280);
    gradient.addColorStop(0, 'rgba(255, 68, 0, 0.2)');
    gradient.addColorStop(1, 'rgba(255, 68, 0, 0.0)');

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Policy Loss',
                data: [],
                borderColor: '#ff4400',
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 4,
                pointHoverBackgroundColor: '#ff4400'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { maxTicksLimit: 10, color: '#606070' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#606070' }
                }
            },
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(20, 20, 35, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#ff4400',
                    borderColor: 'rgba(255, 68, 0, 0.3)',
                    borderWidth: 1,
                    padding: 10
                }
            }
        }
    });
}

/**
 * Create a bar chart for evaluation results.
 */
function createEvalBarChart(canvasId, labels, data, label) {
    var ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    ctx = ctx.getContext('2d');

    var colors = data.map(function (v) {
        return v >= 0 ? 'rgba(0, 240, 255, 0.7)' : 'rgba(255, 68, 0, 0.7)';
    });
    var borderColors = data.map(function (v) {
        return v >= 0 ? '#00f0ff' : '#ff4400';
    });

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label || 'Value',
                data: data,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#606070' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#606070' }
                }
            },
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(20, 20, 35, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#00f0ff',
                    borderColor: 'rgba(0, 240, 255, 0.3)',
                    borderWidth: 1,
                    padding: 10
                }
            }
        }
    });
}

/**
 * Append a data point to a Chart.js chart instance.
 * Keeps a maximum of maxPoints to avoid chart slowdown.
 */
function updateChart(chart, label, value, maxPoints) {
    if (!chart) return;
    maxPoints = maxPoints || 200;
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(value);

    if (chart.data.labels.length > maxPoints) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }

    chart.update('none'); // skip animation for performance
}
