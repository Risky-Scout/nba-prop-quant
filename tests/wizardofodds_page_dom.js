'use strict';

/*
 * A document just real enough to run the player-props page's own script.
 *
 * The page's filter count and the page's visible rows are set by one function,
 * and the requirement is that they can never disagree. Asserting that in
 * Python would mean reimplementing the predicate in the test, which proves
 * only that two copies of a rule agree with each other. So the page's actual
 * script is executed here, against the page's actual markup, with the controls
 * set to whatever the test asked for, and what comes back out is what the
 * script decided: the count it wrote, and which rows it left showing.
 *
 * This models the DOM the script touches and nothing else. It is deliberately
 * not a browser: the stylesheet is not evaluated here, so the half of the
 * requirement that is about CSS is asserted separately, on the stylesheet.
 *
 * Usage: node wizardofodds_page_dom.js <page.html> '<controls as JSON>'
 */

const fs = require('fs');
const vm = require('vm');

function attributesOf(tag) {
  const attributes = {};
  const pattern = /([a-z-]+)(?:="([^"]*)")?/g;
  let found;
  while ((found = pattern.exec(tag)) !== null) {
    attributes[found[1]] = found[2] === undefined ? '' : found[2];
  }
  return attributes;
}

function decode(value) {
  return value
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&amp;/g, '&');
}

function rowsIn(html) {
  const rows = [];
  const pattern = /<tr (data-row[^>]*)>/g;
  let found;
  while ((found = pattern.exec(html)) !== null) {
    const attributes = attributesOf(found[1]);
    rows.push({
      hidden: false,
      style: { display: '' },
      attributes: attributes,
      getAttribute: function (name) {
        return name in attributes ? decode(attributes[name]) : null;
      }
    });
  }
  return rows;
}

function textElement() {
  return { textContent: '', hidden: false, children: [] };
}

function control(value) {
  return { value: value, addEventListener: function () {} };
}

function main() {
  const html = fs.readFileSync(process.argv[2], 'utf8');
  const controls = JSON.parse(process.argv[3] || '{}');

  const rows = rowsIn(html);
  const order = [];

  const elements = {
    'picks-body': {
      appendChild: function (row) {
        order.push(row.getAttribute('data-stable'));
      }
    },
    count: textElement(),
    empty: textElement(),
    pending: textElement(),
    'pending-count': textElement(),
    'f-search': control(controls.search || ''),
    'f-prop': control(controls.prop || 'all'),
    'f-book': control(controls.book || 'all'),
    'f-sort': control(controls.sort || 'ev')
  };

  const document = {
    querySelectorAll: function (selector) {
      if (selector !== 'tr[data-row]') {
        throw new Error('unsupported selector: ' + selector);
      }
      return rows;
    },
    getElementById: function (id) {
      return Object.prototype.hasOwnProperty.call(elements, id)
        ? elements[id]
        : null;
    }
  };

  const script = html.slice(
    html.lastIndexOf('<script>') + '<script>'.length,
    html.lastIndexOf('</script>')
  );

  vm.runInNewContext(script, { document: document });

  const visible = rows.filter(function (row) {
    return !row.hidden && row.style.display !== 'none';
  });

  process.stdout.write(
    JSON.stringify({
      count_text: elements.count.textContent,
      pending_text: elements['pending-count'].textContent,
      empty_hidden: elements.empty.hidden,
      pending_hidden: elements.pending.hidden,
      row_total: rows.length,
      visible: visible.map(function (row) {
        return {
          stable: row.getAttribute('data-stable'),
          offer: row.getAttribute('data-offer'),
          prop: row.getAttribute('data-prop'),
          book: row.getAttribute('data-book'),
          search: row.getAttribute('data-search')
        };
      }),
      order: order
    })
  );
}

main();
