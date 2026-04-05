// Prompt × video VOC heatmap
(function() {
  const container = d3.select('#chart-heatmap');
  if (container.empty()) return;

  const nRows = DATA.heatmap.prompts.length;
  const nCols = DATA.heatmap.videos.length;

  const cellW = 90, cellH = 32;
  const margin = { top: 30, right: 10, bottom: 10, left: 130 };
  const width = nCols * cellW;
  const height = nRows * cellH;
  const fullWidth = width + margin.left + margin.right;

  const svg = container.append('svg')
    .attr('width', fullWidth)
    .attr('height', height + margin.top + margin.bottom);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  // Color scale: diverging, flipped so high VOC = warm
  const colorScale = d3.scaleDiverging()
    .domain([-1, 0.5, 1])
    .interpolator(d3.interpolateRdBu)
    .clamp(true);
  const color = v => colorScale(1 - v);

  // Column headers
  DATA.heatmap.videos.forEach((v, i) => {
    g.append('text')
      .attr('x', i * cellW + cellW/2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '13px')
      .attr('fill', C.text)
      .attr('font-weight', '600')
      .text(v);
  });

  // Row labels
  DATA.heatmap.prompts.forEach((p, i) => {
    g.append('text')
      .attr('x', -6)
      .attr('y', i * cellH + cellH/2)
      .attr('text-anchor', 'end')
      .attr('dy', '0.35em')
      .attr('font-size', '12px')
      .attr('fill', C.muted)
      .text(p);
  });

  // Cells
  DATA.heatmap.values.forEach((row, ri) => {
    row.forEach((val, ci) => {
      g.append('rect')
        .attr('x', ci * cellW + 1)
        .attr('y', ri * cellH + 1)
        .attr('width', cellW - 2)
        .attr('height', cellH - 2)
        .attr('fill', color(val))
        .attr('rx', 3);

      const luminance = d3.lab(color(val)).l;
      g.append('text')
        .attr('x', ci * cellW + cellW/2)
        .attr('y', ri * cellH + cellH/2)
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .attr('font-size', '13px')
        .attr('font-family', 'var(--mono)')
        .attr('fill', luminance < 55 ? 'white' : C.text)
        .text(val.toFixed(2));
    });
  });
})();
