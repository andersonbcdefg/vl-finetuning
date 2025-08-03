// script.js
import { chromium } from "playwright";
import fs from "fs/promises";
import { createWriteStream } from "fs";
import { randomUUID } from "crypto";

// ---------------------------------------------------------------------------
// helpers
function randomDate(start, end) {
  const t = start.getTime() + Math.random() * (end.getTime() - start.getTime());
  return new Date(t);
}
function parts(d) {
  return {
    y: d.getUTCFullYear(),
    m: (d.getUTCMonth() + 1).toString().padStart(2, "0"),
    d: d.getUTCDate().toString().padStart(2, "0"),
  };
}
// ---------------------------------------------------------------------------

(async () => {
  const START = new Date(Date.UTC(2000, 0, 1));
  const END = new Date(Date.UTC(2030, 11, 31, 23));
  const TOTAL = 10_000;
  const CONCURRENCY = 10;

  // generate dates
  const dates = Array.from({ length: TOTAL }, () => randomDate(START, END));

  // disk setup
  await fs.mkdir("screenshots", { recursive: true });
  const csv = createWriteStream("results.csv", { encoding: "utf8" });
  csv.write("isoDate,consoleLogs,screenshotPath\n");

  // browser and persistent tab pool
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const pages = await Promise.all(
    Array.from({ length: CONCURRENCY }, () => context.newPage()),
  );

  // work in batches of <=CONCURRENCY
  for (let batchStart = 0; batchStart < TOTAL; batchStart += CONCURRENCY) {
    const tasks = [];

    for (let i = 0; i < CONCURRENCY && batchStart + i < TOTAL; i++) {
      const idx = batchStart + i;
      const dt = dates[idx];
      const page = pages[i];

      tasks.push(
        (async () => {
          const { y, m, d } = parts(dt);

          // collect console logs just for THIS navigation
          const logs = [];
          const handler = (msg) => logs.push(msg.text());
          page.on("console", handler);

          const url = `https://random-date-coordinates.lovable.app?month=${m}&day=${d}&year=${y}`;
          await page.goto(url, { waitUntil: "load" });

          // unique filename (index + uuid to be absolutely safe)
          const shotPath = `screenshots/${idx.toString().padStart(5, "0")}_${y}-${m}-${d}_${randomUUID()}.jpg`;
          await page.screenshot({
            path: shotPath,
            type: "jpeg",
            quality: 80,
            fullPage: true,
          });

          // write CSV row
          csv.write(
            [
              dt.toISOString(),
              JSON.stringify(logs).replaceAll('"', '""'),
              shotPath,
            ].join(",") + "\n",
          );

          page.off("console", handler); // clean up for next round
        })(),
      );
    }

    await Promise.all(tasks); // wait for this batch
    if ((batchStart + CONCURRENCY) % 1000 === 0)
      // progress each 1000
      console.log(
        `Processed ${Math.min(batchStart + CONCURRENCY, TOTAL)}/${TOTAL}`,
      );
  }

  // tidy up
  await Promise.all(pages.map((p) => p.close()));
  await browser.close();
  csv.end();
})();
