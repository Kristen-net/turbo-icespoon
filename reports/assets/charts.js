(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var danger = style.getPropertyValue('--danger').trim();
  var success = style.getPropertyValue('--success').trim();

  var palette = [accent, accent2, '#8b5cf6', '#f59e0b', '#10b981'];

  // ============ Chart 1: PSNR Training Curve ============
  var psnrChart = echarts.init(document.getElementById('chart-psnr-curve'), null, { renderer: 'svg' });

  // Generate epoch array
  var epochs = [];
  for (var i = 1; i <= 40; i++) epochs.push(i);

  // PSNR data points (from metrics.json, sampled for key points)
  // joint_full: best=18.68
  // no_boxfeat: best=19.23
  // no_uncertainty: best=19.01
  // freeze_backbone: best=19.49

  var psnr_joint = [
    17.16, 18.53, 17.43, 18.41, 18.46, 17.31, 17.47, 18.12, 17.59, 17.16,
    17.55, 17.29, 17.40, 17.10, 17.45, 15.86, 17.43, 17.18, 17.28, 17.34,
    17.36, 17.34, 17.06, 17.24, 17.22, 17.32, 17.43, 17.18, 17.33, 17.41,
    17.20, 17.34, 17.36, 17.39, 17.36, 17.41, 17.38, 17.32, 17.33, 17.34
  ];

  var psnr_nobox = [
    18.53, 18.00, 18.34, 15.73, 17.80, 15.79, 17.52, 17.44, 17.70, 17.31,
    17.31, 17.20, 17.35, 16.99, 16.91, 17.25, 17.07, 17.38, 16.88, 17.41,
    17.36, 17.23, 17.45, 17.38, 16.92, 17.14, 17.43, 17.35, 17.36, 17.43,
    17.19, 17.28, 17.34, 17.37, 17.27, 17.35, 17.36, 17.26, 17.26, 17.28
  ];

  var psnr_nounc = [
    18.45, 18.38, 18.22, 18.03, 18.04, 17.60, 17.52, 17.75, 15.93, 17.33,
    16.95, 17.37, 15.77, 15.18, 17.30, 17.33, 17.41, 17.33, 16.92, 17.33,
    17.35, 17.35, 17.17, 17.37, 17.40, 17.34, 17.34, 17.36, 17.34, 17.43,
    16.98, 17.28, 17.36, 17.37, 17.32, 17.38, 17.36, 17.29, 17.31, 17.32
  ];

  var psnr_freeze = [
    18.79, 18.82, 18.63, 18.63, 18.83, 18.05, 18.67, 18.22, 18.35, 18.65,
    18.07, 18.63, 18.51, 18.49, 18.40, 18.39, 18.42, 18.42, 18.34, 18.35,
    18.37, 18.33, 18.39, 17.66, 17.99, 18.13, 18.25, 18.30, 18.14, 18.28,
    18.25, 18.18, 18.20, 18.20, 18.22, 18.22, 18.20, 18.20, 18.20, 18.20
  ];

  psnrChart.setOption({
    animation: false,
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      valueFormatter: function(v) { return v.toFixed(2) + ' dB'; }
    },
    legend: {
      data: ['joint_full', 'no_boxfeat', 'no_uncertainty', 'freeze_backbone'],
      top: 0,
      textStyle: { color: ink, fontSize: 12 }
    },
    grid: { left: 50, right: 20, top: 40, bottom: 40 },
    xAxis: {
      type: 'category',
      data: epochs,
      name: 'Epoch',
      nameTextStyle: { color: muted },
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      name: 'PSNR (dB)',
      nameTextStyle: { color: muted },
      min: 14.5,
      max: 20,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: 'joint_full',
        type: 'line',
        data: psnr_joint,
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2, color: palette[0] }
      },
      {
        name: 'no_boxfeat',
        type: 'line',
        data: psnr_nobox,
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2, color: palette[1] }
      },
      {
        name: 'no_uncertainty',
        type: 'line',
        data: psnr_nounc,
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2, color: palette[2] }
      },
      {
        name: 'freeze_backbone',
        type: 'line',
        data: psnr_freeze,
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2, color: palette[3] }
      }
    ]
  });

  window.addEventListener('resize', function() { psnrChart.resize(); });

  // ============ Chart 2: Synthetic mAP Bar ============
  var synthChart = echarts.init(document.getElementById('chart-synthetic-map'), null, { renderer: 'svg' });

  synthChart.setOption({
    animation: false,
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      valueFormatter: function(v) { return v.toFixed(3); }
    },
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    xAxis: {
      type: 'category',
      data: ['Hazy\nBaseline', 'no_uncertainty', 'no_boxfeat', 'freeze_backbone', 'joint_full'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11, interval: 0 }
    },
    yAxis: {
      type: 'value',
      name: 'mAP@0.5',
      nameTextStyle: { color: muted },
      min: 0.85,
      max: 1.02,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [{
      type: 'bar',
      data: [
        { value: 1.000, itemStyle: { color: success } },
        { value: 0.997, itemStyle: { color: palette[2] } },
        { value: 0.995, itemStyle: { color: palette[1] } },
        { value: 0.921, itemStyle: { color: palette[3] } },
        { value: 0.915, itemStyle: { color: palette[0] } }
      ],
      barWidth: '50%',
      label: {
        show: true,
        position: 'top',
        formatter: '{c}',
        color: ink,
        fontSize: 11,
        fontWeight: 600
      }
    }]
  });

  window.addEventListener('resize', function() { synthChart.resize(); });

  // ============ Chart 3: Real mAP Bar ============
  var realChart = echarts.init(document.getElementById('chart-real-map'), null, { renderer: 'svg' });

  realChart.setOption({
    animation: false,
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      valueFormatter: function(v) { return v.toFixed(4); }
    },
    grid: { left: 60, right: 20, top: 20, bottom: 40 },
    xAxis: {
      type: 'category',
      data: ['Hazy\nBaseline', 'no_boxfeat', 'no_uncertainty', 'joint_full', 'freeze_backbone'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11, interval: 0 }
    },
    yAxis: {
      type: 'value',
      name: 'mAP@0.5',
      nameTextStyle: { color: muted },
      min: 0,
      max: 0.3,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [{
      type: 'bar',
      data: [
        { value: 0.2508, itemStyle: { color: success } },
        { value: 0.0303, itemStyle: { color: palette[1] } },
        { value: 0.0220, itemStyle: { color: palette[2] } },
        { value: 0.0139, itemStyle: { color: palette[0] } },
        { value: 0.0131, itemStyle: { color: palette[3] } }
      ],
      barWidth: '55%',
      label: {
        show: true,
        position: 'top',
        formatter: '{c}',
        color: ink,
        fontSize: 11,
        fontWeight: 600
      },
      markLine: {
        silent: true,
        data: [{ yAxis: 0.2508, lineStyle: { color: success, type: 'dashed', width: 1 } }],
        label: { show: false }
      }
    }]
  });

  window.addEventListener('resize', function() { realChart.resize(); });

  // ============ Chart 4: Per-class AP ============
  var perClassChart = echarts.init(document.getElementById('chart-per-class'), null, { renderer: 'svg' });

  perClassChart.setOption({
    animation: false,
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      valueFormatter: function(v) { return v.toFixed(3); }
    },
    legend: {
      data: ['Hazy Baseline', 'no_boxfeat', 'no_uncertainty', 'joint_full', 'freeze_backbone'],
      top: 0,
      textStyle: { color: ink, fontSize: 12 }
    },
    grid: { left: 50, right: 20, top: 40, bottom: 40 },
    xAxis: {
      type: 'category',
      data: ['insulator', 'power_line', 'ice', 'tower'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 12 }
    },
    yAxis: {
      type: 'value',
      name: 'AP@0.5',
      nameTextStyle: { color: muted },
      min: 0,
      max: 0.4,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: 'Hazy Baseline',
        type: 'bar',
        data: [0.275, 0.090, 0.352, 0.285],
        itemStyle: { color: success },
        barGap: '10%'
      },
      {
        name: 'no_boxfeat',
        type: 'bar',
        data: [0.010, 0.007, 0.075, 0.030],
        itemStyle: { color: palette[1] }
      },
      {
        name: 'no_uncertainty',
        type: 'bar',
        data: [0.009, 0.013, 0.051, 0.015],
        itemStyle: { color: palette[2] }
      },
      {
        name: 'joint_full',
        type: 'bar',
        data: [0.005, 0.004, 0.027, 0.020],
        itemStyle: { color: palette[0] }
      },
      {
        name: 'freeze_backbone',
        type: 'bar',
        data: [0.007, 0.009, 0.029, 0.007],
        itemStyle: { color: palette[3] }
      }
    ]
  });

  window.addEventListener('resize', function() { perClassChart.resize(); });

  // ============ Chart 5: Radar Chart ============
  var radarChart = echarts.init(document.getElementById('chart-radar'), null, { renderer: 'svg' });

  // Normalize values to 0-100 scale for radar
  // PSNR: 18-20 dB → 0-100
  // Real mAP: 0-0.25 → 0-100
  // Synthetic mAP: 0.9-1.0 → 0-100
  function normPSNR(v) { return Math.max(0, Math.min(100, (v - 18) / 2 * 100)); }
  function normRealMAP(v) { return Math.max(0, Math.min(100, v / 0.25 * 100)); }
  function normSynthMAP(v) { return Math.max(0, Math.min(100, (v - 0.9) / 0.1 * 100)); }

  radarChart.setOption({
    animation: false,
    tooltip: {
      appendToBody: true
    },
    legend: {
      data: ['joint_full', 'no_boxfeat', 'no_uncertainty', 'freeze_backbone'],
      top: 0,
      textStyle: { color: ink, fontSize: 12 }
    },
    radar: {
      indicator: [
        { name: 'PSNR (dB)', max: 100 },
        { name: '真实 mAP', max: 100 },
        { name: '合成 mAP', max: 100 }
      ],
      center: ['50%', '58%'],
      radius: '65%',
      axisName: {
        color: ink,
        fontSize: 13,
        fontWeight: 600
      },
      splitLine: { lineStyle: { color: rule } },
      splitArea: {
        areaStyle: {
          color: ['rgba(0,0,0,0.01)', 'rgba(0,0,0,0.03)']
        }
      },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'radar',
      data: [
        {
          value: [normPSNR(18.68), normRealMAP(0.014), normSynthMAP(0.915)],
          name: 'joint_full',
          lineStyle: { color: palette[0], width: 2 },
          areaStyle: { color: palette[0], opacity: 0.15 },
          itemStyle: { color: palette[0] }
        },
        {
          value: [normPSNR(19.23), normRealMAP(0.030), normSynthMAP(0.995)],
          name: 'no_boxfeat',
          lineStyle: { color: palette[1], width: 2 },
          areaStyle: { color: palette[1], opacity: 0.15 },
          itemStyle: { color: palette[1] }
        },
        {
          value: [normPSNR(19.01), normRealMAP(0.022), normSynthMAP(0.997)],
          name: 'no_uncertainty',
          lineStyle: { color: palette[2], width: 2 },
          areaStyle: { color: palette[2], opacity: 0.15 },
          itemStyle: { color: palette[2] }
        },
        {
          value: [normPSNR(19.49), normRealMAP(0.013), normSynthMAP(0.921)],
          name: 'freeze_backbone',
          lineStyle: { color: palette[3], width: 2 },
          areaStyle: { color: palette[3], opacity: 0.15 },
          itemStyle: { color: palette[3] }
        }
      ]
    }]
  });

  window.addEventListener('resize', function() { radarChart.resize(); });

})();
