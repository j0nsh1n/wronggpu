"""Pastes the sections of out/analysis.txt into RESULTS.md placeholders (MEASURED numbers)."""
import re
a = open("out/analysis.txt").read()
def section(title_prefix, level="## "):
    m = re.search(r"^" + re.escape(level + title_prefix) + r".*?(?=^## |\Z)", a, re.S | re.M)
    return m.group(0).strip() if m else "(missing)"
tables = "\n\n".join(t.strip() for t in re.findall(r"^### .*?(?=^### |^## |\Z)", a, re.S | re.M))
tables = tables.replace("### ", "#### ")
err = "\n".join(l for l in section("Error size").splitlines() if l.startswith("- "))
spec = "\n\n".join(section(t).replace("## ", "#### ", 1) for t in ("Signed zero", "Specials", "FP16 subnormal", "Integer"))
r = open("RESULTS.md").read()
r = r.replace("MODELTABLES", "MEASURED (`out/analysis.txt`). Mismatch = output bits differ (NaN equals NaN); ulp columns = ordered-bit distance "
              "(1 also counts +0 vs -0); rows sorted best first. Non-finite inputs are excluded.\n\n" + tables)
r = r.replace("ERRSIZE", err).replace("SPECIALS", spec)
open("RESULTS.md", "w").write(r)
