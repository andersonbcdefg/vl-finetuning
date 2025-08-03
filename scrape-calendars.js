// script.js
import { chromium } from "playwright";
import fs from "fs/promises";
import { createWriteStream } from "fs";

// ----- helpers -------------------------------------------------------------

function randomDate(start, end) {
  const t = start.getTime() + Math.random() * (end.getTime() - start.getTime());
  return new Date(t);
}

function parts(d) {
  return {
    year: d.getUTCFullYear(),
    month: d.getUTCMonth() + 1, // 1-12
    day: d.getUTCDate(), // 1-31
  };
}

// ----- main ----------------------------------------------------------------

(async () => {
  const START = new Date(Date.UTC(2000, 0, 1)); // 1 Jan 2000
  const END = new Date(Date.UTC(2030, 11, 31, 23)); // 31 Dec 2030
  const COUNT = 10_000;

  // 1. generate the dates
  const dates = Array.from({ length: COUNT }, () => randomDate(START, END));

  // 2. prep disk I/O
  await fs.mkdir("screenshots", { recursive: true });
  const csv = createWriteStream("results.csv", { encoding: "utf8" });
  csv.write("isoDate,consoleLogs,screenshotPath\n");

  // 3. browser session
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();

  for (let i = 0; i < dates.length; i++) {
    const dt = dates[i];
    const { year, month, day } = parts(dt);

    const page = await context.newPage();
    const logs = [];
    page.on("console", (m) => logs.push(m.text()));

    // 4. visit
    const url = `https://random-date-coordinates.lovable.app?month=${month}&day=${day}&year=${year}`;
    await page.goto(url, { waitUntil: "load" });

    // sleep for 1s to allow js to run
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // 5. screenshot
    const shotPath = `screenshots/${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}.jpg`;
    await page.screenshot({
      path: shotPath,
      type: "jpeg",
      quality: 80,
      fullPage: false,
    });

    // 6. record row
    const row =
      [
        dt.toISOString(),
        JSON.stringify(logs).replaceAll('"', '""'), // simple CSV-escaping
        shotPath,
      ].join(",") + "\n";
    csv.write(row);

    await page.close();
    if ((i + 1) % 500 === 0) console.log(`Processed ${i + 1}/${COUNT}`);
  }

  await browser.close();
  csv.end();
})();
