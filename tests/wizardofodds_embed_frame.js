#!/usr/bin/env node
/*
 * Runs the page's iframe height handshake and reports what it broadcast.
 *
 * The handshake is only ever exercised inside someone else's frame, which is
 * the one place a test cannot reach: a browser screenshot of the page shows
 * nothing about it, and a published page that sizes wrong looks identical to
 * one that sizes right until it is embedded. So the script is executed here
 * against a window just real enough to answer the questions that matter --
 * was anything posted, what was posted, and to whom -- with the frame
 * relationship and the measured height supplied by the caller.
 *
 * Usage: node wizardofodds_embed_frame.js <page.html> '<options json>'
 *
 *   framed    false to make the page its own top window, as a reader opening
 *             the published URL directly does
 *   heights   the measured content heights to walk through, in order; the
 *             first is the height at load and each later one is a relayout
 *   wrapperId the element the page is expected to measure
 */

'use strict';

const fs = require('fs');
const vm = require('vm');

const OPENING = '<script id="frame-height">';

function extract(html) {
  const start = html.indexOf(OPENING);
  if (start === -1) {
    throw new Error('the page has no ' + OPENING + ' element');
  }
  return html.slice(start + OPENING.length, html.indexOf('</script>', start));
}

function main() {
  const html = fs.readFileSync(process.argv[2], 'utf8');
  const options = process.argv[3] ? JSON.parse(process.argv[3]) : {};
  const framed = options.framed !== false;
  const heights = options.heights || [420];
  const wrapperId = options.wrapperId || 'app-root';

  let measured = heights[0];

  const posted = [];
  const logged = [];
  const warned = [];
  const observed = [];
  const intervals = [];
  const timeouts = [];
  const listeners = { window: [], document: [] };

  function rect() {
    return { height: measured };
  }

  const wrapper = { id: wrapperId, getBoundingClientRect: rect };
  const body = { getBoundingClientRect: rect };

  const document = {
    body: body,
    get documentElement() {
      return { scrollHeight: measured };
    },
    getElementById: function (id) {
      return id === wrapperId ? wrapper : null;
    },
    addEventListener: function (name, handler) {
      listeners.document.push({ name: name, handler: handler });
    }
  };

  const window = {
    document: document,
    parent: {
      postMessage: function (message, origin) {
        posted.push({ message: message, origin: origin });
      }
    },
    addEventListener: function (name, handler) {
      listeners.window.push({ name: name, handler: handler });
    }
  };

  window.self = window;
  window.top = framed ? { name: 'the embedding page' } : window;

  function MutationObserver(callback) {
    this.observe = function (target, init) {
      observed.push({
        kind: 'mutation',
        isBody: target === body,
        init: init,
        callback: callback
      });
    };
  }

  function ResizeObserver(callback) {
    this.observe = function (target) {
      observed.push({ kind: 'resize', isBody: target === body, callback: callback });
    };
  }

  window.MutationObserver = MutationObserver;
  window.ResizeObserver = ResizeObserver;

  const sandbox = {
    window: window,
    document: document,
    MutationObserver: MutationObserver,
    ResizeObserver: ResizeObserver,
    console: {
      log: function (line) {
        logged.push(String(line));
      },
      warn: function (line, detail) {
        warned.push(String(line) + ' ' + String(detail));
      }
    },
    setInterval: function (handler, ms) {
      intervals.push({ handler: handler, ms: ms });
      return intervals.length;
    },
    setTimeout: function (handler, ms) {
      timeouts.push({ handler: handler, ms: ms });
      return timeouts.length;
    }
  };

  vm.runInNewContext(extract(html), sandbox);

  function fire(where, name) {
    listeners[where]
      .filter(function (entry) {
        return entry.name === name;
      })
      .forEach(function (entry) {
        entry.handler();
      });
  }

  const atLoad = [];

  fire('document', 'DOMContentLoaded');
  fire('window', 'load');
  posted.forEach(function (entry) {
    atLoad.push(entry);
  });

  /* Each later height is a relayout. Driven through the polling callback the
     script registered rather than by calling its function directly, because
     what is being checked is that the page keeps re-measuring on its own. */
  const relayouts = [];

  heights.slice(1).forEach(function (height) {
    const before = posted.length;
    measured = height;
    intervals.forEach(function (entry) {
      entry.handler();
    });
    relayouts.push({
      height: height,
      posted: posted.slice(before)
    });
  });

  /* A click is the page's other trigger: the filters change the row count, so
     the frame has to be told the page got shorter. */
  const clickHandlers = listeners.document.filter(function (entry) {
    return entry.name === 'click';
  });

  process.stdout.write(
    JSON.stringify(
      {
        framed: framed,
        at_load: atLoad,
        relayouts: relayouts,
        posted_total: posted.length,
        origins: posted.map(function (entry) {
          return entry.origin;
        }),
        measured_element_was_wrapper: posted.length > 0,
        logged: logged,
        warned: warned,
        observed: observed.map(function (entry) {
          return { kind: entry.kind, isBody: entry.isBody, init: entry.init };
        }),
        interval_ms: intervals.map(function (entry) {
          return entry.ms;
        }),
        click_handlers: clickHandlers.length,
        click_delay_ms: (function () {
          if (clickHandlers.length === 0) {
            return null;
          }
          const before = timeouts.length;
          clickHandlers[0].handler();
          return timeouts.length > before ? timeouts[before].ms : null;
        })(),
        listener_names: {
          window: listeners.window.map(function (entry) {
            return entry.name;
          }),
          document: listeners.document.map(function (entry) {
            return entry.name;
          })
        }
      },
      null,
      0
    )
  );
}

main();
