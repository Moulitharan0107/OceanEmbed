/**
 * OceanEmbed Frontend Application
 * Interactive map, prediction visualization, and explainability panel
 */

// ============================================================
//  Map Setup — Single, authoritative region lock
// ============================================================

// Exact trained domain: North Indian Ocean 0-25N, 40-100E
const bounds = L.latLngBounds([[0, 40], [25, 100]]);
const paddedBounds = bounds.pad(0.05);

// Create map — zoom IN freely, zoom OUT capped at region boundary
const map = L.map('map', {
    center: [12.5, 70],
    zoom: 5,
    minZoom: 2,  // will be overridden after fitBounds
    maxZoom: 19,
    zoomControl: false,
    worldCopyJump: true,
    scrollWheelZoom: true,
    maxBoundsViscosity: 1.0
});

// Fit to padded bounds and compute minZoom from that
map.fitBounds(paddedBounds);
const fitZoom = map.getZoom();
map.setMinZoom(fitZoom);

// Add zoom control to top-right
L.control.zoom({ position: 'topright' }).addTo(map);

// Dark theme tiles — OpenStreetMap with CSS dark filter (guaranteed free, no key)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
}).addTo(map);

// Prevent panning outside the region
map.on('drag', function() {
    map.panInsideBounds(paddedBounds, { animate: false });
});

console.log('Map initialized — zoom in freely, zoom out capped at region boundary');

// Dotted boundary rectangle for trained domain — dark navy for high contrast
L.rectangle([[0, 40], [25, 100]], {
    color: '#0a1a2f',
    weight: 3,
    fill: false,
    dashArray: '6,6'
}).addTo(map);

// Label for the trained region — dark pill with white text
L.marker([25.8, 70], {
    icon: L.divIcon({
        className: '',
        html: '<div style="background: rgba(10,22,40,0.92); border: 1px solid rgba(0,229,255,0.4); border-radius: 4px; padding: 4px 10px; font-size: 11px; color: #ffffff; white-space: nowrap; font-family: Inter, sans-serif; letter-spacing: 0.3px;">Model Trained: Indian Ocean (-30°–27°N, 32°–120°E) | UI Scoped to North Indian Ocean (0°–25°N, 40°–100°E)</div>',
        iconSize: [0, 0],
        iconAnchor: [-8, 12]
    })
}).addTo(map);

// Markers layer
let currentMarker = null;
let profileData = null;

// Click handler
map.on('click', function(e) {
    const lat = Math.round(e.latlng.lat * 10) / 10;
    const lon = Math.round(e.latlng.lng * 10) / 10;
    
    // Defense: reject clicks outside trained region
    if (lat < 0 || lat > 25 || lon < 40 || lon > 100) {
        alert('Selected point is outside the model\'s trained region (0-25°N, 40-100°E).');
        return;
    }
    
    document.getElementById('inputLat').value = lat;
    document.getElementById('inputLon').value = lon;
    
    runPrediction();
});

// ============================================================
//  Data Status
// ============================================================

async function loadStatus() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');
        
        dot.className = `status-dot ${data.data_mode}`;
        
        if (data.data_mode === 'real') {
            text.textContent = `OceanEmbed · Trained Model`;
        } else if (data.data_mode === 'live') {
            text.textContent = `Live Data · ${data.n_samples} samples`;
        } else if (data.data_mode === 'cached') {
            text.textContent = `OceanEmbed · ${data.n_samples} samples`;
        } else if (data.data_mode === 'synthetic') {
            text.textContent = `OceanEmbed · ${data.n_samples} samples`;
        } else {
            text.textContent = 'Initializing...';
        }
    } catch (e) {
        document.getElementById('statusText').textContent = 'Connection error';
    }
}

// ============================================================
//  Prediction
// ============================================================

async function runPrediction() {
    const lat = parseFloat(document.getElementById('inputLat').value);
    const lon = parseFloat(document.getElementById('inputLon').value);
    const date = document.getElementById('inputDate').value;
    
    if (isNaN(lat) || isNaN(lon)) {
        alert('Please enter valid latitude and longitude values.');
        return;
    }
    
    if (lat < -30 || lat > 30 || lon < 30 || lon > 120) {
        alert('Please select a point within the Indian Ocean domain (30°E–120°E, 30°S–30°N).');
        return;
    }
    
    // Date range validation (model trained 2019-2024, +6 months buffer)
    if (date) {
        const reqDate = new Date(date);
        const maxDate = new Date('2025-06-30');
        if (reqDate > maxDate) {
            alert('Uncertainty will be high — this date is beyond the model\'s validated time range (2019-2024, plus 6 months buffer). Please select a date before 2025-06-30.');
            return;
        }
    }
    
    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Predicting...';
    
    try {
        // Build request
        const body = { latitude: lat, longitude: lon, date: date };
        
        const sstEl = document.getElementById('inputSST');
        const sshEl = document.getElementById('inputSSH');
        const u10El = document.getElementById('inputU10');
        const v10El = document.getElementById('inputV10');
        const sssEl = document.getElementById('inputSSS');
        const curUEl = document.getElementById('inputCurrentU');
        const curVEl = document.getElementById('inputCurrentV');
        
        const sst = sstEl ? sstEl.value : '';
        const ssh = sshEl ? sshEl.value : '';
        const u10 = u10El ? u10El.value : '';
        const v10 = v10El ? v10El.value : '';
        const sss = sssEl ? sssEl.value : '';
        const curU = curUEl ? curUEl.value : '';
        const curV = curVEl ? curVEl.value : '';
        
        if (sst) body.sst = parseFloat(sst);
        if (ssh) body.ssh = parseFloat(ssh);
        if (u10) body.u10 = parseFloat(u10);
        if (v10) body.v10 = parseFloat(v10);
        if (sss) body.sss = parseFloat(sss);
        if (curU) body.current_u = parseFloat(curU);
        if (curV) body.current_v = parseFloat(curV);
        
        const resp = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Prediction failed');
        }
        
        profileData = await resp.json();
        
        // Update map marker
        updateMapMarker(lat, lon, profileData);
        
        // Update location info
        updateLocationInfo(profileData);
        
        // Draw profile chart
        drawProfileChart(profileData);
        
        // Update explainability
        updateExplainability(profileData);
        
        // Show data mode badge
        updateDataModeBadge(profileData);
        
        // Hide hint
        document.getElementById('mapHint').style.display = 'none';
        
    } catch (e) {
        console.error('Prediction error:', e);
        // Friendly message for out-of-bounds errors
        let msg;
        if (e.message.includes('land') || e.message.includes('on land')) {
            msg = 'Selected point is on land. Please select an ocean point to get a subsurface temperature prediction.';
        } else if (e.message.includes('outside') || e.message.includes('region')) {
            msg = 'This model is trained only for the North Indian Ocean region. Please select a point within India\'s surrounding waters.';
        } else if (e.message.includes('Uncertainty') || e.message.includes('beyond') || e.message.includes('date')) {
            msg = 'Uncertainty will be high — this date is beyond the model\'s validated time range (2019-2024, plus 6 months buffer). Please select a date before 2025-06-30.';
        } else {
            msg = `Prediction failed: ${e.message}`;
        }
        alert(msg);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔮 Predict Temperature Profile';
    }
}

// ============================================================
//  Map Marker
// ============================================================

function updateMapMarker(lat, lon, data) {
    if (currentMarker) {
        map.removeLayer(currentMarker);
    }
    
    // Color based on confidence
    const conf = data.confidence_score;
    const color = conf > 0.7 ? '#4CAF50' : conf > 0.4 ? '#FF9800' : '#f44336';
    
    currentMarker = L.circleMarker([lat, lon], {
        radius: 8,
        color: color,
        fillColor: color,
        fillOpacity: 0.8,
        weight: 2,
    }).addTo(map);
    
    // Popup with key info
    const sst = data.predicted_profile[0].toFixed(1);  // 0m = SST
    const t5m = data.predicted_profile[1].toFixed(1);  // 5m = derived
    const shallow = data.predicted_profile[2].toFixed(1);  // 10m = model
    const deep = data.predicted_profile[data.predicted_profile.length - 1].toFixed(1);
    
    currentMarker.bindPopup(`
        <div style="font-family: Inter, sans-serif; font-size: 13px; line-height: 1.5;">
            <strong style="color: #00BCD4;">🌊 OceanEmbed Prediction</strong><br>
            <b>Lat:</b> ${lat.toFixed(2)}° | <b>Lon:</b> ${lon.toFixed(2)}°<br>
            <b>Date:</b> ${data.date}<br>
            <b>SST (0m):</b> ${sst}°C (measured)<br>
            <b>5m:</b> ${t5m}°C (interpolated)<br>
            <b>10m:</b> ${shallow}°C → <b>1000m:</b> ${deep}°C<br>
            <b>Confidence:</b> ${(conf * 100).toFixed(0)}%
        </div>
    `).openPopup();
    
    map.setView([lat, lon], Math.max(map.getZoom(), 5));
}

// ============================================================
//  Location Info Panel
// ============================================================

function updateLocationInfo(data) {
    const infoDiv = document.getElementById('locationInfo');
    infoDiv.style.display = 'grid';
    
    document.getElementById('infoPosition').textContent = 
        `${data.latitude.toFixed(2)}°, ${data.longitude.toFixed(2)}°`;
    document.getElementById('infoDate').textContent = data.date;
    
    // Show features with source info
    const features = data.features_used || {};
    const sst = features.sst ? features.sst.toFixed(1) : 'N/A';
    document.getElementById('infoSST').textContent = `${sst}°C`;
    
    const sss = features.sss ? features.sss.toFixed(1) : 'N/A';
    const sssEl = document.getElementById('infoSSS');
    if (sssEl) sssEl.textContent = `${sss} PSU`;
    
    const conf = data.confidence_score;
    const confEl = document.getElementById('infoConfidence');
    const badge = conf > 0.7 ? 'high' : conf > 0.4 ? 'medium' : 'low';
    confEl.innerHTML = `<span class="confidence-badge ${badge}">${(conf * 100).toFixed(0)}%</span>`;
}

// ============================================================
//  Temperature Profile Chart (Plotly)
// ============================================================

function drawProfileChart(data) {
    const depths = data.depth_levels;
    const temps = data.predicted_profile;
    const uncert = data.uncertainty;
    
    const traces = [];
    
    // Add Argo comparison if available
    if (data.nearest_argo && data.nearest_argo.temperature_profile) {
        const argoDepths = [];
        const argoTemps = [];
        for (const [d, t] of Object.entries(data.nearest_argo.temperature_profile)) {
            argoDepths.push(parseInt(d));
            argoTemps.push(t);
        }
        traces.push({
            x: argoTemps,
            y: argoDepths,
            type: 'scatter',
            mode: 'lines+markers',
            name: `Argo (float ${data.nearest_argo.float_id})`,
            line: { color: '#FF9800', width: 2, dash: 'dot' },
            marker: { size: 5, color: '#FF9800', symbol: 'diamond' },
        });
    }
    
    // Separate derived (0m, 5m) from model-predicted (10m+)
    const derivedTemps = temps.slice(0, 2);
    const derivedDepths = depths.slice(0, 2);
    const modelTemps = temps.slice(2);
    const modelDepths = depths.slice(2);
    const modelUncert = uncert.slice(2);
    
    // Uncertainty band (model-predicted only, 10m-1000m)
    const upper = modelTemps.map((t, i) => t + modelUncert[i]);
    const lower = modelTemps.map((t, i) => t - modelUncert[i]);
    traces.push({
        x: [...upper, ...lower.reverse()],
        y: [...modelDepths, ...modelDepths.slice().reverse()],
        fill: 'toself',
        fillcolor: 'rgba(0, 188, 212, 0.15)',
        line: { color: 'transparent' },
        name: 'Uncertainty',
        showlegend: true,
        hoverinfo: 'skip',
    });
    
    // Model-predicted line (10m-1000m)
    traces.push({
        x: modelTemps,
        y: modelDepths,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted (10-1000m)',
        line: { color: '#00BCD4', width: 2.5 },
        marker: { size: 4, color: '#00BCD4' },
    });
    
    // Derived values (0m, 5m) with different style
    traces.push({
        x: derivedTemps,
        y: derivedDepths,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'SST-anchored (0-5m)',
        line: { color: '#4CAF50', width: 2, dash: 'dash' },
        marker: { size: 6, color: '#4CAF50', symbol: 'circle' },
    });
    
    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#9e9e9e', size: 11 },
        margin: { l: 50, r: 20, t: 10, b: 40 },
        xaxis: {
            title: { text: 'Temperature (°C)', font: { size: 11 } },
            gridcolor: 'rgba(255,255,255,0.05)',
            zeroline: false,
        },
        yaxis: {
            title: { text: 'Depth (m)', font: { size: 11 } },
            gridcolor: 'rgba(255,255,255,0.05)',
            autorange: 'reversed',
            zeroline: false,
        },
        legend: {
            x: 1,
            xanchor: 'right',
            y: 0,
            font: { size: 10 },
        },
        hovermode: 'closest',
    };
    
    Plotly.newPlot('profileChart', traces, layout, {
        responsive: true,
        displayModeBar: false,
    });
}

// ============================================================
//  Explainability Panel
// ============================================================

async function updateExplainability(data) {
    const panel = document.getElementById('explainabilityPanel');
    
    // Get feature values from the prediction input
    const featureMeta = {
        'sst': { label: 'SST', color: '#f44336', desc: 'Sea Surface Temperature' },
        'ssh': { label: 'SSH', color: '#2196F3', desc: 'Sea Surface Height Anomaly' },
        'sss': { label: 'SSS', color: '#4CAF50', desc: 'Sea Surface Salinity' },
        'u10': { label: 'u10 (Wind)', color: '#FF9800', desc: 'Zonal wind component' },
        'v10': { label: 'v10 (Wind)', color: '#FF9800', desc: 'Meridional wind component' },
    };
    
    // Fetch real feature importance from trained model metrics
    let importance = {};
    try {
        const resp = await fetch('/api/metrics');
        if (resp.ok) {
            const metrics = await resp.json();
            importance = metrics.feature_importance || {};
        }
    } catch (e) {
        console.log('Could not fetch feature importance:', e);
    }
    
    // Combine wind components
    const windImp = (importance.u10 || 0) + (importance.v10 || 0);
    const displayImp = {
        'Latitude': importance.latitude || 0,
        'Longitude': importance.longitude || 0,
        'SST': importance.sst || 0,
        'SSH': importance.ssh || 0,
        'Wind (u10+v10)': windImp,
    };
    
    let html = '';
    
    for (const [name, imp] of Object.entries(displayImp)) {
        const pct = (imp * 100).toFixed(0);
        const color = name === 'SST' ? '#f44336' : name === 'SSH' ? '#2196F3' : name.includes('Wind') ? '#FF9800' : name === 'Latitude' ? '#9C27B0' : '#607D8B';
        html += `
            <div class="feature-bar-container">
                <div class="feature-bar-label">
                    <span class="feature-bar-name">${name}</span>
                    <span class="feature-bar-value">${pct}%</span>
                </div>
                <div class="feature-bar-track">
                    <div class="feature-bar-fill" style="width: ${pct}%; background: ${color};"></div>
                </div>
            </div>
        `;
    }
    
    html += `
        <div style="margin-top: 10px; font-size: 0.75em; color: var(--text-muted); line-height: 1.4;">
            <strong>Inputs:</strong> Latitude, Longitude, SST, SSH, U10, V10<br>
            <strong>Note:</strong> SSS unavailable (SMOS 403 Forbidden)<br>
            <strong>Method:</strong> Gradient-based feature attribution on trained model.
        </div>
    `;
    
    panel.innerHTML = html;
}

// ============================================================
//  Data Mode Badge
// ============================================================

function updateDataModeBadge(data) {
    const badge = document.getElementById('dataModeBadge');
    if (!badge) return;
    
    const mode = data.data_mode || 'unknown';
    const features = data.features_used || {};
    
    let label, color;
    if (mode === 'real') {
        label = 'REAL SURFACE DATA';
        color = '#4CAF50';
    } else if (mode === 'partial') {
        label = 'PARTIAL REAL DATA';
        color = '#FF9800';
    } else {
        label = 'ESTIMATED INPUTS';
        color = '#f44336';
    }
    
    badge.innerHTML = `<span style="color: ${color}; font-weight: 600;">${label}</span>`;
    
    // Update feature status
    const featureList = document.getElementById('featureStatusList');
    if (featureList) {
        const sources = data.features_used ? {} : {};
        // Sources aren't in predict response yet - use feature values to infer
        featureList.innerHTML = `
            <div style="font-size: 0.75em; color: var(--text-muted); margin-top: 4px;">
                SST: ${features.sst?.toFixed(1)}°C | SSH: ${features.ssh?.toFixed(4)}m | 
                U10: ${features.u10?.toFixed(2)}m/s | V10: ${features.v10?.toFixed(2)}m/s<br>
                SSS: NOT USED (SMOS unavailable)
            </div>
        `;
    }
}



// ============================================================
//  Initialize
// ============================================================

loadStatus();
setInterval(loadStatus, 30000);

document.getElementById('inputDate').value = '2022-06-15';
