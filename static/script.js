// ============================================================
// CrackSeg V2 — Frontend Logic
// ============================================================

// --- DOM Elements ---
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadContent = document.getElementById('upload-content');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const downloadBtn = document.getElementById('download-btn');

const emptyState = document.getElementById('empty-state');
const loadingState = document.getElementById('loading-state');
const resultsDisplay = document.getElementById('results-display');
const statusBadge = document.getElementById('status-badge');

const resultOverlay = document.getElementById('result-overlay');
const resultMask = document.getElementById('result-mask');
const resultHeatmap = document.getElementById('result-heatmap');
const resultOriginal = document.getElementById('result-original');

const tabs = document.querySelectorAll('.tab');
const resultViews = document.querySelectorAll('.result-view');

const aboutModal = document.getElementById('about-modal');
const navAbout = document.getElementById('nav-about');
const closeModal = document.getElementById('close-modal');

let currentFile = null;
let lastAnalysisData = null;

// ============================================================
// 1. DRAG & DROP / FILE SELECT
// ============================================================

dropZone.addEventListener('click', () => {
    if (!currentFile) fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev =>
    dropZone.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); }, false)
);

['dragenter', 'dragover'].forEach(ev =>
    dropZone.addEventListener(ev, () => dropZone.classList.add('dragover'), false)
);

['dragleave', 'drop'].forEach(ev =>
    dropZone.addEventListener(ev, () => dropZone.classList.remove('dragover'), false)
);

dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file.type.match('image.*')) {
        alert('Please select an image file (JPG, PNG).');
        return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadContent.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;
        resetResults();
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    previewContainer.classList.add('hidden');
    uploadContent.classList.remove('hidden');
    analyzeBtn.disabled = true;
    resetResults();
});

// ============================================================
// 2. API CALL
// ============================================================

analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    emptyState.classList.add('hidden');
    resultsDisplay.classList.add('hidden');
    loadingState.classList.remove('hidden');

    statusBadge.textContent = 'Processing...';
    statusBadge.className = 'status-badge processing';

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();

        if (response.ok && data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Server error occurred');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze image: ' + error.message);
        resetResults();
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span>Analyze Image</span><i class="fa-solid fa-arrow-right"></i>';
    }
});

// ============================================================
// 3. DISPLAY RESULTS
// ============================================================

function displayResults(data) {
    lastAnalysisData = data;

    // Set images
    resultOriginal.src = data.original;
    resultOverlay.src = data.overlay;
    resultMask.src = data.mask;
    if (data.heatmap) resultHeatmap.src = data.heatmap;

    // Stat cards
    document.getElementById('stat-resolution').textContent = data.resolution || '--';
    document.getElementById('stat-confidence').textContent = data.confidence || '--';
    document.getElementById('stat-num-cracks').textContent = data.num_cracks !== undefined ? data.num_cracks : '--';
    document.getElementById('stat-max-length').textContent = data.max_crack_length_px ? data.max_crack_length_px + ' px' : '--';
    document.getElementById('stat-avg-width').textContent = data.avg_width_px ? data.avg_width_px + ' px' : '--';
    document.getElementById('stat-crack-ratio').textContent = data.crack_ratio || '--';

    // Integrity Gauge
    animateGauge(data.integrity_raw || 0, data.health_color);

    // Health Banner
    const healthBanner = document.getElementById('health-banner');
    const statHealth = document.getElementById('stat-health');
    const healthScoreText = document.getElementById('health-score-text');

    statHealth.textContent = data.health_status;
    healthScoreText.textContent = data.integrity_score || '--';

    healthBanner.classList.remove('health-green', 'health-yellow', 'health-orange', 'health-red');
    healthBanner.classList.add(`health-${data.health_color}`);

    // Show results
    loadingState.classList.add('hidden');
    resultsDisplay.classList.remove('hidden');
    resultsDisplay.classList.remove('animate-in');
    void resultsDisplay.offsetWidth; // force reflow
    resultsDisplay.classList.add('animate-in');

    statusBadge.textContent = 'Complete';
    statusBadge.className = 'status-badge success';

    downloadBtn.disabled = false;
    switchTab('overlay-view');
}

// ============================================================
// 4. CIRCULAR GAUGE ANIMATION
// ============================================================

function animateGauge(value, color) {
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeValue = document.getElementById('gauge-value');
    const circumference = 2 * Math.PI * 52; // r=52

    const colorMap = {
        green:  '#34d399',
        yellow: '#fbbf24',
        orange: '#fb923c',
        red:    '#f87171'
    };

    const offset = circumference - (value / 100) * circumference;
    gaugeFill.style.strokeDashoffset = offset;
    gaugeFill.style.stroke = colorMap[color] || '#6366f1';

    // Animated counter
    let current = 0;
    const step = Math.max(1, Math.round(value / 40));
    const counter = setInterval(() => {
        current += step;
        if (current >= value) {
            current = value;
            clearInterval(counter);
        }
        gaugeValue.textContent = current;
        gaugeValue.style.color = colorMap[color] || '#f1f5f9';
    }, 25);
}

// ============================================================
// 5. TABS
// ============================================================

tabs.forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.getAttribute('data-target')));
});

function switchTab(targetId) {
    tabs.forEach(t => {
        t.classList.toggle('active', t.getAttribute('data-target') === targetId);
    });
    resultViews.forEach(v => {
        v.classList.toggle('active', v.id === targetId);
    });
}

// ============================================================
// 6. RESET
// ============================================================

function resetResults() {
    emptyState.classList.remove('hidden');
    loadingState.classList.add('hidden');
    resultsDisplay.classList.add('hidden');
    statusBadge.textContent = 'Waiting';
    statusBadge.className = 'status-badge';
    downloadBtn.disabled = true;
    lastAnalysisData = null;

    // Reset gauge
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeValue = document.getElementById('gauge-value');
    gaugeFill.style.strokeDashoffset = 326.73;
    gaugeValue.textContent = '--';
}

// ============================================================
// 7. DOWNLOAD REPORT
// ============================================================

downloadBtn.addEventListener('click', () => {
    if (!lastAnalysisData) return;
    const d = lastAnalysisData;

    const report = `
╔══════════════════════════════════════════════════════════════╗
║              CRACKSEG — STRUCTURAL HEALTH REPORT            ║
╚══════════════════════════════════════════════════════════════╝

  Date:              ${new Date().toLocaleString()}
  Model:             Custom U-Net (PyTorch)
  Original Resolution: ${d.resolution}

──────────────────────────────────────────────────────────────
  STRUCTURAL INTEGRITY SCORE:   ${d.integrity_score}
  HEALTH STATUS:                ${d.health_status}
──────────────────────────────────────────────────────────────

  ANALYSIS FACTORS:
  ├── Crack Area Coverage:      ${d.crack_ratio}
  ├── Number of Cracks:         ${d.num_cracks}
  ├── Maximum Crack Length:     ${d.max_crack_length_px} px
  ├── Average Crack Width:      ${d.avg_width_px} px
  └── Model Confidence:         ${d.confidence}

  FACTOR SCORES (damage contribution):
  ├── Area Score:               ${d.factors?.area || 'N/A'} / 25
  ├── Count Score:              ${d.factors?.count || 'N/A'} / 20
  ├── Length Score:              ${d.factors?.length || 'N/A'} / 25
  ├── Width Score:               ${d.factors?.width || 'N/A'} / 15
  └── Branching Score:           ${d.factors?.branching || 'N/A'} / 15

──────────────────────────────────────────────────────────────
  Generated by CrackSeg | Shubham Kumar | IIT Bombay SoS 2026
──────────────────────────────────────────────────────────────
`.trim();

    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `CrackSeg_Report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
});

// ============================================================
// 8. ABOUT MODAL
// ============================================================

navAbout.addEventListener('click', (e) => {
    e.preventDefault();
    aboutModal.classList.remove('hidden');
});

closeModal.addEventListener('click', () => {
    aboutModal.classList.add('hidden');
});

aboutModal.addEventListener('click', (e) => {
    if (e.target === aboutModal) aboutModal.classList.add('hidden');
});
