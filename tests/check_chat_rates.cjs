// Run with node tests/check_chat_rates.cjs. Executes the actual inline helpers.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../ui/chat.html'), 'utf8');
for (const [, script] of html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)) new vm.Script(script);
const source = html.slice(html.indexOf('        class RateSeries {'), html.indexOf('        function createRateChart('));
const context = vm.createContext({});
vm.runInContext(source + '\nglobalThis.api = { RateSeries, rateQuantile, rateDistribution, rateShareAtLeast, rateDensity, paintRateDistribution, paintRateTail };', context);
const { RateSeries, rateQuantile, rateDistribution, rateShareAtLeast, rateDensity, paintRateDistribution, paintRateTail } = context.api;
const samples = values => values.map(rate => ({ rate, duration: 1 }));
assert.equal(rateQuantile([], .5), null);
assert.equal(rateQuantile([7], .9999), 7);
for (const p of [.5, .8, .9, .95, .99, .999, .9999]) {
    assert.ok(Math.abs(rateQuantile([0, 10, 20, 30, 40], p) - p * 40) < 1e-10);
}
assert.equal(rateDistribution([]), null);
assert.equal(rateDistribution([{ rate: 1000, duration: .001 }]), null);
assert.equal(rateDistribution([...samples([0, 10]), { rate: 1000, duration: .01 }]).sorted.length, 2);
assert.equal(rateShareAtLeast([], 20), null);
assert.equal(rateShareAtLeast([0, 20, 20, 40], 20), .75);
assert.equal(rateShareAtLeast([0, 20, 20, 40], 21), .25);
assert.equal(rateShareAtLeast([0, 20, 20, 40], 41), 0);
assert.equal(rateShareAtLeast([0, 20, 20, 40], 0), 1);
const shift = rateDistribution(samples([...Array(30).fill(10), ...Array(30).fill(40)]));
assert.equal(shift.previous.length, 30);
assert.equal(shift.recent.length, 30);
assert.equal(rateShareAtLeast(shift.previous, 20), 0);
assert.equal(rateShareAtLeast(shift.recent, 20), 1);
assert.equal(rateDistribution(samples(Array(10).fill(5))).previous.length, 0);
const outlier = rateDistribution(samples([1, 2, 3, 4, 100]));
assert.equal(outlier.low, 1); assert.equal(outlier.high, 4); assert.equal(outlier.outliers[0], 100);
for (const rates of [[0], [20, 20], [0, 0, 1, 15, 40, 100]]) {
    const density = rateDensity(rates, Math.max(1, ...rates) * 1.08);
    assert.ok(density.every(v => Number.isFinite(v) && v >= 0 && v <= 1));
}
const series = new RateSeries();
series.observe(1000, 'a', '', 10);
series.observe(2000, 'a', '', 20);
series.observe(4100, 'a', '', 30);
assert.equal(series.samples(4100)[0].rate, 20);
assert.equal(series.samples(4100)[1].rate, 0); // Pauses count.
assert.equal(rateDistribution(series.samples(4100)).sorted.length, 3);
series.finish(4100);
assert.equal(series.samples(10000).length, 4); // No idle windows after completion.
const estimated = new RateSeries();
estimated.observe(0, 'abcd', '', undefined);
estimated.observe(1000, 'efgh', '', undefined);
estimated.observe(1100, '', '', 900); // Final-only usage must not invent a spike.
assert.equal(estimated.count, 2);
assert.equal(estimated.source, 'estimate');

// Exercise SVG geometry and labels, including empty/constant distributions.
function chartStub() {
    const elements = new Map();
    return { querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, { value: '20', textContent: '', innerHTML: '',
            setAttribute(name, value) { this[name] = value; } });
        return elements.get(selector);
    }, elements };
}
for (const values of [[], [0], [20], [0, 0, 0, 100], Array(60).fill(20)]) {
    const chart = chartStub(), state = { chart, rateSeries: { source: 'estimate' }, end: null };
    paintRateDistribution(state, samples(values));
    state.end = 2000; paintRateDistribution(state, samples(values));
    for (const el of chart.elements.values()) assert.ok(!/NaN|Infinity/.test(el.innerHTML));
    if (values.length) {
        assert.ok(chart.querySelector('[data-percentile="99.99"]').textContent.startsWith('≈'));
        chart.querySelector('.rate-threshold').value = '0';
        paintRateTail(chart, chart.rateDistribution);
        assert.ok(chart.querySelector('.rate-tail-result').textContent.startsWith('100.0%'));
        chart.querySelector('.rate-threshold').value = '';
        paintRateTail(chart, chart.rateDistribution);
        assert.match(chart.querySelector('.rate-tail-result').textContent, /Enter a nonnegative/);
    }
}
console.log('PASS: script syntax, percentiles, window eligibility, pauses, rolling frequency, outliers, density, and SVG rendering');
