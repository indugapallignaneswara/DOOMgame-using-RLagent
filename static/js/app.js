/* ============================================================
   DOOM RL – Main Application JS
   ============================================================ */

(function () {
    'use strict';

    // ---- SocketIO Connection ----
    const socket = io({ transports: ['websocket', 'polling'] });
    window.socket = socket;

    const connectionDot    = document.getElementById('connectionDot');
    const connectionStatus = document.getElementById('connectionStatus');

    socket.on('connect', function () {
        if (connectionDot) connectionDot.style.background = 'var(--success)';
        if (connectionStatus) connectionStatus.textContent = 'Connected';
        console.log('[DOOM RL] Socket connected:', socket.id);
    });

    socket.on('disconnect', function () {
        if (connectionDot) connectionDot.style.background = 'var(--danger)';
        if (connectionStatus) connectionStatus.textContent = 'Disconnected';
        console.warn('[DOOM RL] Socket disconnected');
    });

    socket.on('connect_error', function (err) {
        if (connectionDot) connectionDot.style.background = 'var(--warning)';
        if (connectionStatus) connectionStatus.textContent = 'Connection Error';
        console.error('[DOOM RL] Connection error:', err.message);
    });

    socket.on('error', function (data) {
        window.showToast(data.message || 'An error occurred', 'error');
    });

    // ---- Navigation Highlighting ----
    (function highlightNav() {
        const path = window.location.pathname;
        const links = document.querySelectorAll('#sidebarNav a');
        links.forEach(function (link) {
            link.classList.remove('active');
            const href = link.getAttribute('href');
            if (href === path || (href !== '/' && path.startsWith(href))) {
                link.classList.add('active');
            } else if (href === '/' && path === '/') {
                link.classList.add('active');
            }
        });
    })();

    // ---- Mobile Menu ----
    var menuToggle = document.getElementById('menuToggle');
    var sidebar = document.getElementById('sidebar');
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function () {
            sidebar.classList.toggle('open');
        });
        // Close on outside click
        document.addEventListener('click', function (e) {
            if (sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) &&
                !menuToggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
    }

    // ---- Utility Functions ----
    window.formatNumber = function (num, decimals) {
        if (num === null || num === undefined || isNaN(num)) return '--';
        decimals = decimals !== undefined ? decimals : 2;
        if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(1) + 'M';
        if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(1) + 'K';
        return parseFloat(num).toFixed(decimals);
    };

    window.formatTime = function (seconds) {
        if (!seconds || seconds < 0) return '0s';
        var h = Math.floor(seconds / 3600);
        var m = Math.floor((seconds % 3600) / 60);
        var s = Math.floor(seconds % 60);
        if (h > 0) return h + 'h ' + m + 'm';
        if (m > 0) return m + 'm ' + s + 's';
        return s + 's';
    };

    // ---- Toast Notification System ----
    window.showToast = function (message, type) {
        type = type || 'info';
        var container = document.getElementById('flashMessages');
        if (!container) return;

        var iconMap = {
            success: 'check-circle',
            error:   'exclamation-circle',
            warning: 'exclamation-triangle',
            info:    'info-circle'
        };

        var toast = document.createElement('div');
        toast.className = 'flash flash-' + type;
        toast.innerHTML = '<i class="fas fa-' + (iconMap[type] || 'info-circle') + '"></i> ' + message;
        container.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(function () {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(40px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(function () {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 300);
        }, 5000);
    };

    // ---- Loading Overlay Helpers ----
    window.showLoading = function (text) {
        var overlay = document.getElementById('loadingOverlay');
        var loadText = document.getElementById('loadingText');
        if (overlay) overlay.classList.add('visible');
        if (loadText && text) loadText.textContent = text;
    };

    window.hideLoading = function () {
        var overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.classList.remove('visible');
    };

    // ---- Auto-dismiss flash messages ----
    var flashes = document.querySelectorAll('.flash');
    flashes.forEach(function (flash) {
        setTimeout(function () {
            flash.style.opacity = '0';
            flash.style.transform = 'translateX(40px)';
            flash.style.transition = 'all 0.3s ease';
            setTimeout(function () {
                if (flash.parentNode) flash.parentNode.removeChild(flash);
            }, 300);
        }, 5000);
    });

})();
