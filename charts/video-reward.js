// Video + reward curve widget (Chart.js)
// Shows NORMALIZED (0-1) reward curves so ensemble visually appears above 2B
(function() {
  const container = document.getElementById('video-reward-widget');
  if (!container) return;

  const videoSrc = container.dataset.video || 'data/videos/fold_towel.mp4';

  container.innerHTML = `
    <div class="vrw-layout">
      <div class="vrw-video-section">
        <video class="vrw-video" muted playsinline preload="auto" autoplay loop>
          <source src="${videoSrc}" type="video/mp4" />
        </video>
      </div>
      <div class="vrw-chart-section">
        <canvas class="vrw-canvas"></canvas>
      </div>
    </div>
    <div class="vrw-controls">
      <button class="vrw-play-btn">Pause</button>
    </div>
  `;

  const video = container.querySelector('.vrw-video');
  const canvas = container.querySelector('.vrw-canvas');
  const btn = container.querySelector('.vrw-play-btn');

  // Normalize each method's raw log-probs to [0, 1]
  function normalize(arr) {
    const mn = Math.min(...arr), mx = Math.max(...arr);
    if (mx === mn) return arr.map(() => 0.5);
    return arr.map(v => (v - mn) / (mx - mn));
  }

  const raw = DATA.curves.fold_towel;
  const methods = [
    { label: '8B baseline',       color: C.b8,  data: normalize(raw.b8) },
    { label: '2B baseline',       color: C.b2,  data: normalize(raw.b2) },
    { label: '3-prompt ensemble', color: C.ens, data: normalize(raw.ens) },
  ];

  const N = 10;

  // Interpolate for smooth drawing
  function interpolateData(fullData, progress) {
    const maxIdx = (N - 1) * progress;
    const result = [];
    for (let i = 0; i <= N - 1; i++) {
      if (i <= maxIdx) {
        result.push({ x: i + 1, y: fullData[i] });
      } else if (i === Math.ceil(maxIdx) && maxIdx > 0) {
        const frac = maxIdx - Math.floor(maxIdx);
        const prev = fullData[Math.floor(maxIdx)];
        const next = fullData[i];
        result.push({ x: maxIdx + 1, y: prev + frac * (next - prev) });
        break;
      } else {
        break;
      }
    }
    return result;
  }

  function initChart() {
    const vLinePlugin = {
      id: 'vLine',
      afterDraw(chart) {
        const t = chart._videoT || 0;
        if (t <= 0) return;
        const { left, right, top, bottom } = chart.chartArea;
        const x = left + t * (right - left);
        const ctx = chart.ctx;
        ctx.save();
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = 'rgba(0,0,0,0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.stroke();
        ctx.restore();
      }
    };

    const datasets = methods.map(m => ({
      label: m.label,
      borderColor: m.color,
      backgroundColor: m.color + '12',
      data: [],
      _full: m.data,
      tension: 0.3,
      pointRadius: 3,
      pointBackgroundColor: m.color,
      borderWidth: 2,
      fill: true,
    }));

    const chart = new Chart(canvas, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            type: 'linear', min: 1, max: N,
            title: { display: true, text: 'Video prefix', font: { size: 11 } },
            ticks: { stepSize: 1, font: { size: 10 } },
            grid: { color: '#f0f0e8' },
          },
          y: {
            min: -0.05, max: 1.05,
            title: { display: true, text: 'Normalized progress', font: { size: 11 } },
            ticks: { font: { size: 10 } },
            grid: { color: '#f0f0e8' },
          }
        },
        plugins: {
          legend: {
            position: 'top',
            labels: { font: { size: 11 }, boxWidth: 12, padding: 12 }
          }
        }
      },
      plugins: [vLinePlugin]
    });

    chart._videoT = 0;
    return chart;
  }

  let chart = null;

  video.addEventListener('loadedmetadata', () => {
    if (typeof Chart !== 'undefined' && !chart) {
      chart = initChart();
    }
  });

  let lastT = -1;
  function tick() {
    if (chart && !video.paused && !video.ended && video.duration) {
      const t = video.currentTime / video.duration;
      if (Math.abs(t - lastT) > 0.001) {
        chart._videoT = t;
        chart.data.datasets.forEach(ds => {
          ds.data = interpolateData(ds._full, t);
        });
        chart.update('none');
        lastT = t;
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  video.addEventListener('ended', () => {
    if (chart) {
      chart._videoT = 1;
      chart.data.datasets.forEach(ds => {
        ds.data = ds._full.map((v, i) => ({ x: i + 1, y: v }));
      });
      chart.update('none');
    }
  });

  btn.addEventListener('click', () => {
    if (video.paused) {
      video.play();
      btn.textContent = 'Pause';
    } else {
      video.pause();
      btn.textContent = 'Play';
    }
  });
})();
