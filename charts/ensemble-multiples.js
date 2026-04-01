// Small multiples: 4 videos × 3 methods reward curves
(function() {
  const container = d3.select('#chart-ensemble-multiples');
  if (container.empty()) return;

  const panels = [
    { key: 'fold_towel', title: 'Fold towel' },
    { key: 'put_pen_cup', title: 'Pen → cup' },
    { key: 'remove_cap', title: 'Remove cap' },
    { key: 'put_block_cup', title: 'Block → cup' },
  ];

  const N = 10;

  panels.forEach(panel => {
    const div = container.append('div');
    const voc = DATA.curves[panel.key].voc;
    div.append('div').attr('class', 'panel-title').text(
      `${panel.title}  (${voc.b2.toFixed(2)} / ${voc.ens.toFixed(2)} / ${voc.b8.toFixed(2)})`
    );

    const cw = 400, ch = 180;
    const margin = { top: 10, right: 10, bottom: 25, left: 40 };
    const w = cw - margin.left - margin.right;
    const h = ch - margin.top - margin.bottom;

    const svg = div.append('svg')
      .attr('viewBox', `0 0 ${cw} ${ch}`)
      .attr('width', '100%');
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const d = DATA.curves[panel.key];
    const allVals = [...d.b2, ...d.ens, ...d.b8];
    const yMin = Math.min(...allVals), yMax = Math.max(...allVals);

    const x = d3.scaleLinear().domain([1, N]).range([0, w]);
    const y = d3.scaleLinear().domain([yMin - 0.3, yMax + 0.3]).range([h, 0]);

    g.append('g').attr('class', 'axis')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(0));
    g.append('g').attr('class', 'axis')
      .call(d3.axisLeft(y).ticks(4).tickSize(-w))
      .selectAll('.tick line').attr('stroke', '#f0f0e8');

    const line = d3.line().x((d,i) => x(i+1)).y(v => y(v)).curve(d3.curveMonotoneX);

    const series = [
      { data: d.b8, color: C.b8, width: 1.5 },
      { data: d.b2, color: C.b2, width: 1.5 },
      { data: d.ens, color: C.ens, width: 2.5 },
    ];

    series.forEach(s => {
      g.append('path').datum(s.data)
        .attr('fill', 'none')
        .attr('stroke', s.color)
        .attr('stroke-width', s.width)
        .attr('d', line);
      g.selectAll(null).data(s.data).enter().append('circle')
        .attr('cx', (v,i) => x(i+1))
        .attr('cy', v => y(v))
        .attr('r', 2.5)
        .attr('fill', s.color);
    });
  });
})();
