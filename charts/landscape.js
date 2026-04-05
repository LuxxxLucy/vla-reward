// Multi-line chart: reward-scope multi-method comparison on stackcubes
(function() {
  const container = d3.select('#chart-landscape');
  if (container.empty()) return;

  const fullWidth = Math.min(container.node().clientWidth, 900);
  const margin = { top: 15, right: 130, bottom: 35, left: 45 };
  const width = fullWidth - margin.left - margin.right;
  const height = 260;

  const svg = container.append('svg')
    .attr('width', fullWidth)
    .attr('height', height + margin.top + margin.bottom);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([1, 10]).range([0, width]);
  const y = d3.scaleLinear().domain([0, 1.05]).range([height, 0]);

  // Axes
  g.append('g').attr('class', 'axis')
    .attr('transform', `translate(0,${height})`)
    .call(d3.axisBottom(x).ticks(10).tickSize(0));
  g.append('text').attr('x', width/2).attr('y', height + 30)
    .attr('text-anchor', 'middle').attr('font-size', '13px').attr('fill', C.muted)
    .text('Video prefix');

  g.append('g').attr('class', 'axis')
    .call(d3.axisLeft(y).ticks(5).tickSize(-width).tickFormat(d3.format('.1f')))
    .selectAll('.tick line').attr('stroke', '#f0f0e8');

  const line = d3.line()
    .x((d,i) => x(i+1))
    .y(d => y(d))
    .curve(d3.curveMonotoneX);

  // Draw lines
  DATA.landscape.methods.forEach(m => {
    g.append('path')
      .datum(m.values)
      .attr('fill', 'none')
      .attr('stroke', m.color)
      .attr('stroke-width', m.name.startsWith('Our') ? 2.5 : 1.5)
      .attr('opacity', m.name.startsWith('Our') ? 1 : 0.6)
      .attr('stroke-dasharray', m.name === 'GVL' ? '4,3' : 'none')
      .attr('d', line);
  });

  // End-of-line labels with collision avoidance
  const labels = DATA.landscape.methods.map(m => ({
    name: m.name,
    color: m.color,
    y: y(m.values[m.values.length - 1])
  }));
  labels.sort((a, b) => a.y - b.y);
  const minGap = 12;
  for (let i = 1; i < labels.length; i++) {
    if (labels[i].y - labels[i-1].y < minGap) {
      labels[i].y = labels[i-1].y + minGap;
    }
  }
  labels.forEach(l => {
    g.append('text')
      .attr('x', width + 6)
      .attr('y', l.y)
      .attr('dy', '0.35em')
      .attr('font-size', '12px')
      .attr('fill', l.color)
      .text(l.name);
  });
})();
