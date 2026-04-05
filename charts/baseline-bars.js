// Vertical grouped bar chart: 2B vs ensemble vs 8B for 4 tasks
(function() {
  const container = d3.select('#chart-baseline');
  if (container.empty()) return;

  const tasks = [
    DATA.baseline.find(d => d.video === 'Fold towel'),
    DATA.baseline.find(d => d.video === 'Pen → cup'),
    DATA.baseline.find(d => d.video === 'Remove cap'),
    DATA.baseline.find(d => d.video === 'Block → cup'),
  ];

  const methods = [
    { key: 'b2',  color: C.b2,  label: '2B' },
    { key: 'ens', color: C.ens, label: 'Ensemble' },
    { key: 'b8',  color: C.b8,  label: '8B' },
  ];

  const fullWidth = Math.min(container.node().clientWidth, 520);
  const margin = { top: 25, right: 10, bottom: 50, left: 45 };
  const width = fullWidth - margin.left - margin.right;
  const height = 260;

  const svg = container.append('svg')
    .attr('width', fullWidth)
    .attr('height', height + margin.top + margin.bottom);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x0 = d3.scaleBand().domain(tasks.map(d => d.video)).range([0, width]).padding(0.25);
  const x1 = d3.scaleBand().domain(methods.map(m => m.key)).range([0, x0.bandwidth()]).padding(0.06);
  const y = d3.scaleLinear().domain([0, 1.05]).range([height, 0]);

  // Y axis
  g.append('g').attr('class', 'axis')
    .call(d3.axisLeft(y).ticks(5).tickSize(-width).tickFormat(d3.format('.1f')))
    .selectAll('.tick line').attr('stroke', '#f0f0e8');

  // X axis
  g.append('g').attr('class', 'axis')
    .attr('transform', `translate(0,${height})`)
    .call(d3.axisBottom(x0).tickSize(0))
    .selectAll('text')
      .attr('font-size', '13px')
      .attr('transform', 'rotate(-15)')
      .attr('text-anchor', 'end');

  // Bars
  tasks.forEach(d => {
    methods.forEach(m => {
      const xPos = x0(d.video) + x1(m.key);
      const barH = height - y(d[m.key]);

      g.append('rect')
        .attr('x', xPos)
        .attr('y', y(d[m.key]))
        .attr('width', x1.bandwidth())
        .attr('height', barH)
        .attr('fill', m.color)
        .attr('rx', 2)
        .attr('opacity', 0.85);

      // Value label on top
      g.append('text')
        .attr('x', xPos + x1.bandwidth() / 2)
        .attr('y', y(d[m.key]) - 4)
        .attr('text-anchor', 'middle')
        .attr('font-size', '9px')
        .attr('font-family', 'var(--mono)')
        .attr('fill', C.muted)
        .text(d[m.key].toFixed(2));
    });
  });

  // Legend
  const legend = g.append('g').attr('transform', `translate(0, -12)`);
  methods.forEach((m, i) => {
    const lx = i * 90;
    legend.append('rect').attr('x', lx).attr('y', 0).attr('width', 10).attr('height', 10)
      .attr('fill', m.color).attr('rx', 2);
    legend.append('text').attr('x', lx + 14).attr('y', 9)
      .attr('font-size', '13px').attr('fill', C.muted).text(m.label);
  });
})();
