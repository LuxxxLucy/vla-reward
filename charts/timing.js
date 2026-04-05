// Timing: horizontal bars, relative cost (2B cached single = 1×)
// Vision/language split from actual experiment data where available
(function() {
  const container = d3.select('#chart-timing');
  if (container.empty()) return;

  const t = DATA.timing;

  // All rows use real data where we have it
  const rows = [
    {
      label: '2B single (cached)',
      total: t.cached_single_total,
      vision: t.cached_single_vision,
      language: t.cached_single_language,
      showSplit: true,
      color: C.b2
    },
    {
      label: '2B ensemble (cached)',
      total: t.cached_ens3_total,
      vision: t.cached_ens3_vision,
      language: t.cached_ens3_language,
      showSplit: true,
      color: C.ens
    },
    {
      label: '2B ensemble ×10 (cached)',
      total: t.cached_ens10_total,
      vision: t.cached_ens10_vision,
      language: t.cached_ens10_language,
      showSplit: true,
      color: C.ens
    },
    {
      label: '8B single',
      total: t.b8_total,
      vision: t.b8_vision,
      language: t.b8_language,
      showSplit: true,
      color: C.b8
    },
  ];

  // Normalize to first row (2B single cached)
  const base = rows[0].total;
  rows.forEach(r => { r.mult = r.total / base; });

  const fullWidth = Math.min(container.node().clientWidth, 520);
  const margin = { top: 15, right: 55, bottom: 30, left: 155 };
  const width = fullWidth - margin.left - margin.right;
  const height = rows.length * 42;

  const svg = container.append('svg')
    .attr('width', fullWidth)
    .attr('height', height + margin.top + margin.bottom);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const maxMult = Math.max(...rows.map(r => r.mult));
  const x = d3.scaleLinear().domain([0, maxMult * 1.08]).range([0, width]);
  const y = d3.scaleBand().domain(rows.map(d => d.label)).range([0, height]).padding(0.3);

  g.append('g').attr('class', 'axis')
    .attr('transform', `translate(0,${height})`)
    .call(d3.axisBottom(x).ticks(4).tickFormat(d => d.toFixed(0) + '×').tickSize(0));

  g.append('g').attr('class', 'axis')
    .call(d3.axisLeft(y).tickSize(0))
    .selectAll('text').attr('font-size', '13px');

  rows.forEach(d => {
    const yPos = y(d.label);
    const bh = y.bandwidth();
    const totalW = x(d.mult);

    if (d.showSplit && d.vision !== null) {
      const visFrac = d.vision / d.total;
      const visW = totalW * visFrac;
      const langW = totalW * (1 - visFrac);

      g.append('rect')
        .attr('x', 0).attr('y', yPos)
        .attr('width', visW).attr('height', bh)
        .attr('fill', C.vision).attr('rx', 2);

      g.append('rect')
        .attr('x', visW).attr('y', yPos)
        .attr('width', langW).attr('height', bh)
        .attr('fill', C.language).attr('rx', 2);

      if (visW > 20) {
        g.append('text')
          .attr('x', visW / 2).attr('y', yPos + bh / 2)
          .attr('text-anchor', 'middle').attr('dy', '0.35em')
          .attr('font-size', '9px').attr('fill', 'white').text('vis');
      }
      if (langW > 20) {
        g.append('text')
          .attr('x', visW + langW / 2).attr('y', yPos + bh / 2)
          .attr('text-anchor', 'middle').attr('dy', '0.35em')
          .attr('font-size', '9px').attr('fill', 'white').text('lang');
      }
    } else {
      g.append('rect')
        .attr('x', 0).attr('y', yPos)
        .attr('width', totalW).attr('height', bh)
        .attr('fill', d.color).attr('rx', 2)
        .attr('opacity', d.label.includes('naive') ? 0.4 : 0.6);
    }

    g.append('text')
      .attr('x', totalW + 5).attr('y', yPos + bh / 2)
      .attr('dy', '0.35em')
      .attr('font-size', '13px').attr('fill', '#111')
      .attr('font-weight', d.showSplit ? '600' : '400')
      .text(d.mult.toFixed(1) + '×');
  });

  const legend = g.append('g').attr('transform', `translate(0, -8)`);
  [{ label: 'Vision', color: C.vision }, { label: 'Language', color: C.language }].forEach((l, i) => {
    legend.append('rect').attr('x', i * 80).attr('y', 0).attr('width', 10).attr('height', 10).attr('fill', l.color).attr('rx', 2);
    legend.append('text').attr('x', i * 80 + 14).attr('y', 9).attr('font-size', '12px').attr('fill', '#6b7280').text(l.label);
  });
})();
