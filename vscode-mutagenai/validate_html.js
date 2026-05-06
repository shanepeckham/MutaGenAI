const fs = require("fs");
const path = require("path");
const html = fs.readFileSync(path.join(process.env.TMPDIR || "/tmp", "test_webview.html"), "utf8");
const m = html.match(/<script[^>]*>([\s\S]*?)<\/script>/);
if (!m) { console.log("NO SCRIPT TAG FOUND"); process.exit(1); }
const js = m[1];
console.log("Script length:", js.length, "chars");
console.log("First 300 chars:", js.substring(0, 300));
try {
  new Function(js);
  console.log("JS SYNTAX OK");
} catch(e) {
  console.log("JS SYNTAX ERROR:", e.message);
}
