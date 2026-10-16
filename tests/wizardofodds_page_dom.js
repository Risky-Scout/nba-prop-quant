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
 * script decided: the count it wrote, which rows it left showing, which
 * distribution panels it left open, and which odds format it settled on.
 *
 * This models the DOM the script touches and nothing else. It is deliberately
 * not a browser: the stylesheet is not evaluated here, so the half of the
 * requirement that is about CSS is asserted separately, on the stylesheet.
 * That matters most for the odds format, where the script's whole job is to
 * set one attribute and the showing and hiding is the stylesheet's -- so what
 * this harness can prove about it is that the attribute lands, the control
 * agrees with it, and storage is read and written and never anything else.
 *
 * Usage: node wizardofodds_page_dom.js <page.html> '<controls as JSON>'
 *
 * Controls: search, prop, book, sort, odds, stored, storage ("blocked" to
 * model a browser that refuses to read or write it), click (a list of panel
 * ids whose buttons are pressed, in order, after the first render).
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

function element(tag) {
  const attributes = attributesOf(tag);
  return {
    hidden: false,
    style: { display: '' },
    attributes: attributes,
    id: attributes.id === undefined ? '' : decode(attributes.id),
    listeners: {},
    getAttribute: function (name) {
      return name in attributes ? decode(attributes[name]) : null;
    },
    setAttribute: function (name, value) {
      attributes[name] = String(value);
    },
    addEventListener: function (name, handler) {
      this.listeners[name] = handler;
    }
  };
}

/* Every <tr> the page renders, in document order, each tagged with whether it
   is a prediction or the panel belonging to one. The script finds predictions
   by selector and panels by id, so both have to be here and both have to be
   the same objects the script mutates. */
function rowsIn(html) {
  const rows = [];
  const pattern = /<tr ([^>]*)>/g;
  let found;
  while ((found = pattern.exec(html)) !== null) {
    const made = element(found[1]);
    if (made.getAttribute('data-row') !== null) {
      made.kind = 'prediction';
      rows.push(made);
    } else if ((made.getAttribute('class') || '').indexOf('wf-dist-row') !== -1) {
      made.kind = 'panel';
      rows.push(made);
    }
  }
  return rows;
}

function togglesIn(html) {
  const buttons = [];
  const pattern = /<button ([^>]*data-dist-toggle[^>]*)>/g;
  let found;
  while ((found = pattern.exec(html)) !== null) {
    buttons.push(element(found[1]));
  }
  return buttons;
}

function textElement() {
  return {
    textContent: '',
    hidden: false,
    children: [],
    addEventListener: function () {}
  };
}

function control(value) {
  return {
    value: value,
    listeners: {},
    addEventListener: function (name, handler) {
      this.listeners[name] = handler;
    }
  };
}

function storage(controls) {
  if (controls.storage === 'blocked') {
    return {
      reads: 0,
      writes: [],
      getItem: function () {
        this.reads += 1;
        throw new Error('storage is blocked in this browser');
      },
      setItem: function () {
        this.writes.push(null);
        throw new Error('storage is blocked in this browser');
      }
    };
  }

  const held = {};
  if (controls.stored !== undefined) {
    held[controls.stored_key || 'wf.nba.playerProps.oddsFormat'] =
      controls.stored;
  }
  return {
    reads: 0,
    writes: [],
    held: held,
    getItem: function (key) {
      this.reads += 1;
      return Object.prototype.hasOwnProperty.call(held, key)
        ? held[key]
        : null;
    },
    setItem: function (key, value) {
      this.writes.push([key, value]);
      held[key] = value;
    }
  };
}

function main() {
  const html = fs.readFileSync(process.argv[2], 'utf8');
  const controls = JSON.parse(process.argv[3] || '{}');

  const everyRow = rowsIn(html);
  const rows = everyRow.filter(function (row) {
    return row.kind === 'prediction';
  });
  const toggles = togglesIn(html);
  /* ``order`` is the predictions, in the order the script put them back, and
     ``placed`` is everything it appended -- so a test can assert both that
     the predictions are sorted and that each panel landed immediately after
     the prediction it belongs to. */
  const order = [];
  const placed = [];

  const wrapper = element('id="app-root" data-odds-format="american"');

  const elements = {
    'app-root': wrapper,
    'picks-body': {
      appendChild: function (row) {
        if (row.kind === 'panel') {
          placed.push({ panel: row.id });
          return;
        }
        order.push(row.getAttribute('data-stable'));
        placed.push({ row: row.getAttribute('data-stable') });
      }
    },
    count: textElement(),
    empty: textElement(),
    pending: textElement(),
    'pending-count': textElement(),
    'f-search': control(controls.search || ''),
    'f-prop': control(controls.prop || 'all'),
    'f-book': control(controls.book || 'all'),
    'f-sort': control(controls.sort || 'ev'),
    'f-odds': control(controls.odds || 'american')
  };

  everyRow.forEach(function (row) {
    if (row.id) {
      elements[row.id] = row;
    }
  });

  const document = {
    querySelectorAll: function (selector) {
      if (selector === 'tr[data-row]') {
        return rows;
      }
      if (selector === '[data-dist-toggle]') {
        return toggles;
      }
      throw new Error('unsupported selector: ' + selector);
    },
    querySelector: function (selector) {
      const found = /^\[aria-controls="([^"]+)"\]$/.exec(selector);
      if (!found) {
        throw new Error('unsupported selector: ' + selector);
      }
      const matching = toggles.filter(function (button) {
        return button.getAttribute('aria-controls') === found[1];
      });
      return matching.length ? matching[0] : null;
    },
    getElementById: function (id) {
      return Object.prototype.hasOwnProperty.call(elements, id)
        ? elements[id]
        : null;
    }
  };

  const localStorage = storage(controls);

  /* By id, not "the last script on the page": the page also carries the site's
     iframe height handshake, and running that one here would prove nothing
     about the filtering this harness exists to check. */
  const opening = '<script id="page-controls">';
  const start = html.indexOf(opening);
  if (start === -1) {
    throw new Error('the page has no ' + opening + ' element');
  }
  const script = html.slice(
    start + opening.length,
    html.indexOf('</script>', start)
  );

  const window = { localStorage: localStorage };

  vm.runInNewContext(script, { document: document, window: window });

  /* Buttons pressed after the first render, so a test can assert what one
     click does rather than only what the page settles at. */
  (controls.click || []).forEach(function (id) {
    toggles.forEach(function (button) {
      if (button.getAttribute('aria-controls') === id && button.listeners.click) {
        button.listeners.click();
      }
    });
  });

  if (controls.choose !== undefined) {
    elements['f-odds'].value = controls.choose;
    if (elements['f-odds'].listeners.change) {
      elements['f-odds'].listeners.change();
    }
  }

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
      odds_format: wrapper.getAttribute('data-odds-format'),
      odds_control: elements['f-odds'].value,
      storage_reads: localStorage.reads,
      storage_writes: localStorage.writes,
      toggles: toggles.map(function (button) {
        return {
          panel: button.getAttribute('aria-controls'),
          expanded: button.getAttribute('aria-expanded'),
          hidden: button.hidden
        };
      }),
      panels: everyRow
        .filter(function (row) {
          return row.kind === 'panel';
        })
        .map(function (row) {
          return {
            id: row.id,
            hidden: row.hidden,
            display: row.style.display
          };
        }),
      visible: visible.map(function (row) {
        return {
          stable: row.getAttribute('data-stable'),
          offer: row.getAttribute('data-offer'),
          prop: row.getAttribute('data-prop'),
          book: row.getAttribute('data-book'),
          search: row.getAttribute('data-search')
        };
      }),
      order: order,
      placed: placed
    })
  );
}

main();
