#!/usr/bin/env python3
"""Build the domain SFT mixture for the coding fine-tune.

Domains: general coding, Linux/shell, networking, WiFi, IoT/embedded, and
security (defensive + authorized offensive/pentest methodology).

Three constraints shape this file:

1. LICENSE. Training-data licenses propagate to the tuned weights, so only
   MIT / Apache-2.0 / CC-BY / CC0 / ODC-BY sources are used. Deliberately
   excluded despite being obvious hits:
     - nickrosh/Evol-Instruct-Code-80k-v1  (CC-BY-NC-SA, non-commercial)
     - baig31/Cybersecurity_penetration_testing_books (scraped books)

2. "RED TEAM" IS AMBIGUOUS ON THE HUB. Most datasets with that phrase are LLM
   jailbreak/adversarial-prompt corpora (walledai/TDC23-RedTeaming,
   CohereLabs/aya_redteaming, 9mark9/llm-redteam-owasp-prompts, ...). Those
   teach nothing about security engineering and would degrade the model's
   behaviour. Only security-*engineering* sources are included here.

3. NO WIFI OR IOT INSTRUCTION DATASETS EXIST. Every Hub hit for "wifi" is
   signal data (CSI/RSSI); every "IoT" hit is network-traffic capture for
   intrusion detection. g4lihru/arduino-dataset is the sole real embedded-code
   source. The WiFi slice is therefore mined by keyword and will stay small --
   it is reported honestly rather than padded, because upsampling ~150 examples
   to look balanced just memorizes those 150.

HELD OUT (never trained on): preemware/pentesting-eval -- the domain benchmark.

    python scripts/build_sft_dataset.py --out ~/data/sft_mix
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re

GENERAL_CODE = [
    ("ise-uiuc/Magicoder-OSS-Instruct-75K",             "mit"),
    ("ise-uiuc/Magicoder-Evol-Instruct-110K",           "apache-2.0"),
    ("m-a-p/CodeFeedback-Filtered-Instruction",         "apache-2.0"),
    ("glaiveai/glaive-code-assistant",                  "apache-2.0"),
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k",   "odc-by"),
    ("sahil2801/CodeAlpaca-20k",                        "cc-by-4.0"),
    ("TokenBender/code_instructions_122k_alpaca_style", "apache-2.0"),
]
# (repo, kwargs for load_dataset) -- several repos need an explicit data file.
LINUX_SRC = [
    ("dilkushsingh/NL2Bash", {"data_files": "NL2bash_train.csv"}),
]
SECURITY_SRC = [
    ("me-aas/pentesting-explanations",              {}),
    ("darkknight25/KALI_LINUX_TOOLKIT_DATASET",     {}),
    ("AYI-NEDJIMI/bug-bounty-pentest-en",           {}),
    ("0xn0cta/bugbounty-hunter-v1",                 {}),
    ("CyberNative/Code_Vulnerability_Security_DPO", {}),
    ("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", {}),
    ("AYI-NEDJIMI/mitre-attack-en", {"data_files": "data/qa_dataset.json"}),
]

# DROPPED, with reasons -- kept here so the decision is auditable:
#   7h3-R3v3n4n7/pentest-agent-dataset-alpaca
#       287k rows of CVE-database dumps behind a "[general_info]" prefix, many
#       with no content at all ("** RESERVED ** This candidate has been
#       reserved..."). Training on it teaches recitation of text the model
#       cannot memorize, plus the placeholder string itself.
#   g4lihru/arduino-dataset
#       ships dataset.bin + tokenizer/ -- pre-tokenized PRETRAINING data, not
#       instruction pairs. Unusable for SFT.
#   missvector/linux-commands
#       multilingual translations of command descriptions; marginal value.
#   stasvinokur/cve-and-cwe-*, lemon42-ai/*, ethanolivertroy/nist-*
#       classification tables and RAG chunks, not instruction pairs.

# VM / virtualization / infrastructure. New domain in v2.
VM_SRC = [
    ("MattCoddity/dockerNLcommands",              {}),
    ("stindardlogic/devops-kubernetes-sft-100k",  {}),
    ("galcan/terraform_sec",                      {}),
    ("bernabepuente/devops-cloud-instruction-dataset", {}),
]

# Extra security instruction sources (v2).
SECURITY_SRC_V2 = [
    ("rezaduty/cybersecurity-qa-v2", {}),
]

# Structured rows that need hand-written formatting into Q/A.
STRUCTURED_SEC = [
    ("darkknight25/RED_team_tactics_dataset",
     {"data_files": "RED_team_tactics_dataset.jsonl"}, "format_red_team"),
    ("darkknight25/APT_STYLE_Privilege_Escalation_Dataset", {}, "format_privesc"),
    ("darkknight25/blue_team_defense_dataset", {}, "format_blueteam"),
    ("darkknight25/Networking_Commands_Dataset", {}, "format_netcmd"),
    ("darkknight25/Cloud_Vulnerabilities_Dataset", {}, "format_cloudvuln"),
    ("darkknight25/Shellcode_Exploit_Dataset", {}, "format_shellcode"),
]


def _lst(v):
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def format_privesc(row: dict):
    """Priv-esc technique -> Q/A, always tagged as authorized-testing only."""
    desc, cmd = row.get("description"), row.get("command")
    cat = row.get("category") or "privilege escalation"
    tac = row.get("tactics") or ""
    if not desc or not cmd:
        return None
    q = (f"During an authorized penetration test, explain the '{cat}' "
         f"privilege-escalation technique and show the command.")
    a = (f"**{cat}**"
         + (f" — {tac}" if tac else "") + f"\n\n{desc}\n\n"
         f"```bash\n{cmd}\n```\n\n"
         f"Only run this against systems you are explicitly authorized to "
         f"test. Defenders should monitor for this pattern and apply least "
         f"privilege to prevent it.")
    return q, a


def format_blueteam(row: dict):
    """Defensive detection rule."""
    threat, sig = row.get("threat"), row.get("signature")
    rtype = row.get("rule_type") or "detection"
    tool = row.get("tool") or ""
    tech = row.get("mapped_technique") or ""
    if not threat or not sig:
        return None
    q = (f"Write a {rtype} detection rule for '{threat}'"
         + (f" (MITRE {tech})" if tech else "") + ".")
    a = (f"**{threat}** — {rtype} rule"
         + (f" for {tool}" if tool else "") + ":\n\n```\n{}\n```".format(sig)
         + (f"\n\nMaps to MITRE technique {tech}." if tech else ""))
    return q, a


def format_netcmd(row: dict):
    """Networking/security command with its built-in risk note."""
    cmd, desc = row.get("command"), row.get("description")
    usage = row.get("usage") or ""
    cat = row.get("category") or "networking"
    risk = row.get("risk") or ""
    if not cmd or not desc:
        return None
    q = f"How do I {desc[0].lower() + desc[1:].rstrip('.')} from the command line?"
    a = f"**{cat}**\n\n```bash\n{cmd}\n```\n\n{desc}"
    if usage:
        a += f"\n\nUsage: {usage}"
    if risk:
        a += f"\n\n⚠ {risk}"
    return q, a


def format_cloudvuln(row: dict):
    desc, poc = row.get("description"), row.get("poc")
    cat = row.get("category") or "misconfiguration"
    prov = row.get("cloud_provider") or "cloud"
    vcode = row.get("vulnerable_code") or ""
    if not desc:
        return None
    q = (f"Explain this {prov} {cat} vulnerability and how to test for it "
         f"safely: {desc}")
    parts = [f"**{prov} {cat}**\n\n{desc}"]
    if vcode:
        parts.append(f"Vulnerable configuration:\n```\n{str(vcode)[:900]}\n```")
    if poc:
        parts.append(f"Proof of concept (authorized testing only):\n"
                     f"```bash\n{poc}\n```")
    parts.append("Remediation: apply least-privilege access and audit the "
                 "resource's exposure.")
    return q, "\n\n".join(parts)


def format_shellcode(row: dict):
    """Exploit-dev learning example. Emphasizes the analysis, not raw bytes."""
    desc = row.get("description")
    vt = row.get("vulnerability_type") or "exploit"
    plat = row.get("platform") or ""
    arch = row.get("architecture") or ""
    goal = row.get("payload_goal") or ""
    cve = row.get("cve") or ""
    if not desc:
        return None
    q = (f"In an authorized exploit-development exercise, explain a {vt} "
         f"targeting {plat}/{arch}"
         + (f" ({cve})" if cve and cve.lower() != "n/a" else "") + ".")
    a = (f"**{vt}** on {plat}/{arch}\n\n{desc}\n\n"
         f"Goal: {goal}\n\n"
         f"This is for authorized testing and defensive research only. "
         f"Mitigations: enable DEP/NX, ASLR, and stack canaries; keep the "
         f"affected software patched.")
    return q, a


def format_red_team(row: dict) -> tuple[str, str] | None:
    """Turn a structured ATT&CK tactic row into a Q/A pair.

    Mitigations are always included in the answer: the source pairs each
    technique with its defence, and keeping them together is what makes this
    security-engineering material rather than a recipe list.
    """
    name = row.get("tactic_name")
    tech = row.get("mitre_technique")
    desc = row.get("description")
    if not (name and desc):
        return None
    steps = row.get("execution_steps") or []
    tools = row.get("tools") or []
    mitig = row.get("mitigations") or []
    if isinstance(steps, str):
        steps = [steps]
    if isinstance(tools, str):
        tools = [tools]
    if isinstance(mitig, str):
        mitig = [mitig]
    if not mitig:
        return None  # no defence half -> skip

    q = (f"Explain the {name} technique ({tech}) in an authorized penetration "
         f"test: how it works, and how a defender detects and mitigates it.")
    parts = [f"**{name}** ({tech})\n\n{desc}\n"]
    if steps:
        parts.append("**How it is executed in an engagement:**\n" +
                     "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
    if tools:
        parts.append("**Common tooling:** " + ", ".join(map(str, tools)))
    parts.append("**Detection and mitigation:**\n" +
                 "\n".join(f"- {m}" for m in mitig))
    if row.get("impact"):
        parts.append(f"**Impact if successful:** {row['impact']}")
    return q, "\n\n".join(parts)

IN_KEYS = ["problem", "instruction", "query", "question", "nl", "prompt",
           "input", "task", "command_description", "description", "title",
           "user", "text"]
OUT_KEYS = ["solution", "response", "answer", "output", "chosen", "bash",
            "completion", "command", "code", "explanation", "assistant"]

KW = {
    "wifi": [r"\bwi-?fi\b", r"802\.11", r"\bwpa[23]?\b", r"\bwep\b", r"hostapd",
             r"wpa_supplicant", r"\bssid\b", r"\bbssid\b", r"\brssi\b",
             r"monitor mode", r"\bnl80211\b", r"iwconfig", r"\biwlist\b",
             r"\bairmon\b", r"\baircrack\b", r"beacon frame", r"\bwlan\d?\b",
             r"access point", r"\bdeauth", r"\bpmkid\b", r"\beap-?ol\b",
             r"wireless (network|adapter|interface|card)", r"\bwifi\b"],
    "iot": [r"\bmqtt\b", r"\besp32\b", r"\besp8266\b", r"\barduino\b",
            r"raspberry ?pi", r"\bgpio\b", r"\bi2c\b", r"\bspi bus\b",
            r"\bmodbus\b", r"\bzigbee\b", r"\bcoap\b", r"micropython",
            r"platformio", r"\bfirmware\b", r"\bembedded c\b", r"\brtos\b",
            r"\bservo\b", r"\bdht11\b", r"\bads1115\b", r"\buart\b",
            r"digitalwrite", r"analogread",
            r"\bstm32\b", r"\bavr\b", r"\bplatform ?io\b", r"sensor (read|data)"],
    # NOTE: \bsetup\(\) and \bloop\(\) were removed -- they matched every
    # XCTest/unittest setUp() in the corpus and flooded IoT with unit-testing
    # questions.
    "networking": [r"\bsocket\b", r"\btcp\b", r"\budp\b", r"\bdns\b",
                   r"\bsubnet", r"netmiko", r"paramiko", r"\bscapy\b",
                   r"\bpacket\b", r"tcpdump", r"wireshark", r"\biptables\b",
                   r"netfilter", r"\bgrpc\b", r"\bipv[46]\b", r"port scan",
                   r"\bnetcat\b", r"\bsnmp\b", r"routing table", r"\bnmap\b",
                   r"\bvlan\b", r"\bbgp\b", r"\bospf\b", r"\bproxy\b"],
    "linux": [r"\bbash\b", r"shell script", r"\bsystemd\b", r"\bcrontab\b",
              r"\bchmod\b", r"\bawk\b", r"\bsed\b", r"/proc/", r"/etc/",
              r"kernel module", r"apt-get", r"\bdmesg\b", r"\bstrace\b",
              r"\bsudo\b", r"\bmakefile\b", r"\bssh\b", r"\bgrep\b"],
    "security": [r"\bvulnerab", r"\bsanitiz", r"sql injection", r"\bxss\b",
                 r"\bcsrf\b", r"\bbcrypt\b", r"\bhashlib\b", r"\btls\b",
                 r"certificate", r"authenticat", r"cve-\d", r"\bcwe-\d",
                 r"\bfuzz", r"input validation", r"\bowasp\b",
                 r"privilege escalation", r"secure cod", r"\bencrypt",
                 r"hardening", r"least privilege", r"\bpentest", r"\bexploit\b",
                 r"\bpayload\b", r"\bmitre\b", r"\bmalware\b"],
}
KW_RE = {k: re.compile("|".join(v), re.I) for k, v in KW.items()}


def pick(row: dict, keys: list[str]) -> str | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def from_conversation(row: dict) -> tuple[str, str] | None:
    """Handle ShareGPT / chatml style rows."""
    for key in ("conversations", "messages", "conversation"):
        conv = row.get(key)
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        user = assistant = None
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from") or turn.get("role") or "").lower()
            content = turn.get("value") or turn.get("content")
            if not isinstance(content, str):
                continue
            if role in ("human", "user") and user is None:
                user = content.strip()
            elif role in ("gpt", "assistant") and user is not None:
                assistant = content.strip()
                break
        if user and assistant:
            return user, assistant
    return None


def normalize(row: dict) -> tuple[str, str] | None:
    got = from_conversation(row)
    if got:
        return got
    q = pick(row, IN_KEYS)
    a = pick(row, OUT_KEYS)
    if not q or not a or q == a:
        return None
    extra = row.get("input")
    if isinstance(extra, str) and extra.strip() and extra.strip() != q:
        q = f"{q}\n\n{extra.strip()}"
    return q, a


def classify(q: str, a: str) -> str:
    blob = f"{q}\n{a}"
    # wifi/iot are matched against the INSTRUCTION only. Matching the full
    # blob produced false positives like a mobile-app design question that
    # merely mentioned wireless in passing -- the topic has to be what was
    # asked, not something the answer wandered into.
    for dom in ("wifi", "iot"):
        if KW_RE[dom].search(q):
            return dom
    for dom in ("networking", "security", "linux"):
        if KW_RE[dom].search(blob):
            return dom
    return "code"


# Rows matching any of these are dropped regardless of source: placeholder
# text, template prefixes, and stubs that teach the model to emit filler.
JUNK_RE = re.compile(
    r"\*\* *RESERVED *\*\*|candidate has been reserved|"
    r"^\s*\[[a-z_]+\]\s|rejected reason|no description available|"
    r"^\s*(n/?a|none|unknown|tbd)\s*$", re.I)


def is_junk(q: str, r: str) -> bool:
    if JUNK_RE.search(q) or JUNK_RE.search(r):
        return True
    if len(r) < 80:          # too short to teach anything
        return True
    if r.strip().lower() == q.strip().lower():
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-domain", type=int, default=3500)
    ap.add_argument("--general-code", type=int, default=6000)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    from datasets import Dataset, load_dataset

    rng = random.Random(a.seed)
    DOMAINS = ("code", "linux", "networking", "wifi", "iot", "vm", "security")
    buckets: dict[str, list[dict]] = {k: [] for k in DOMAINS}
    seen: set[str] = set()

    FORMATTERS = {
        "format_red_team": format_red_team,
        "format_privesc": format_privesc,
        "format_blueteam": format_blueteam,
        "format_netcmd": format_netcmd,
        "format_cloudvuln": format_cloudvuln,
        "format_shellcode": format_shellcode,
    }

    dropped = {"junk": 0, "dupe": 0, "len": 0}

    def add(q: str, r: str, dom: str, src: str) -> bool:
        if not (a.min_chars <= len(q) + len(r) <= a.max_chars):
            dropped["len"] += 1
            return False
        if is_junk(q, r):
            dropped["junk"] += 1
            return False
        h = hashlib.sha1(q[:400].lower().encode()).hexdigest()
        if h in seen:
            dropped["dupe"] += 1
            return False
        seen.add(h)
        buckets[dom].append({"instruction": q, "response": r,
                             "domain": dom, "source": src})
        return True

    def ingest(repo: str, kwargs: dict | None = None,
               force_dom: str | None = None, formatter=None) -> None:
        kwargs = dict(kwargs or {})
        try:
            ds = load_dataset(repo, split="train", **kwargs)
        except Exception:
            try:
                d = load_dataset(repo, **kwargs)
                ds = d[list(d.keys())[0]]
            except Exception as e:
                print(f"  SKIP {repo:<52} {type(e).__name__}")
                return
        n = 0
        for row in ds:
            got = formatter(row) if formatter else normalize(row)
            if not got:
                continue
            q, r = got
            dom = force_dom or classify(q, r)
            n += add(q, r, dom, repo.split("/")[-1])
        flag = ""
        if n == 0:
            flag = f"   <- 0 rows; columns were {list(ds.column_names)[:8]}"
        print(f"  {repo:<52} +{n}{flag}")

    print("=== general code (classified by keyword) ===")
    for repo, _ in GENERAL_CODE:
        ingest(repo)
    print("=== linux / shell ===")
    for repo, kw in LINUX_SRC:
        ingest(repo, kw, force_dom="linux")
    print("=== vm / virtualization / infra ===")
    for repo, kw in VM_SRC:
        ingest(repo, kw, force_dom="vm")
    print("=== security (engineering, not LLM red-teaming) ===")
    for repo, kw in SECURITY_SRC + SECURITY_SRC_V2:
        ingest(repo, kw, force_dom="security")
    print("=== security (structured -> hand-formatted) ===")
    for repo, kw, fname in STRUCTURED_SEC:
        ingest(repo, kw, force_dom="security", formatter=FORMATTERS[fname])

    print(f"\n  filtered out: {dropped['junk']} junk, {dropped['dupe']} dupes, "
          f"{dropped['len']} length")

    print("\n=== pool sizes before sampling ===")
    for k in DOMAINS:
        print(f"  {k:<12} {len(buckets[k])}")

    rows: list[dict] = []
    for dom in DOMAINS:
        cap = a.general_code if dom == "code" else a.per_domain
        pool = buckets[dom]
        rng.shuffle(pool)
        rows.extend(pool[:cap])
    rng.shuffle(rows)

    print("\n=== final mixture ===")
    final: dict[str, int] = {}
    for r in rows:
        final[r["domain"]] = final.get(r["domain"], 0) + 1
    for k in sorted(final):
        print(f"  {k:<12} {final[k]:>6}  ({100 * final[k] / len(rows):5.1f}%)")
    print(f"  {'TOTAL':<12} {len(rows):>6}")

    thin = [k for k, v in final.items() if v < 500]
    if thin:
        print(f"\n  NOTE: under-represented domains: {thin}")
        print("  Reported as-is. Upsampling these would memorize a few hundred")
        print("  examples rather than teach the domain.")

    out = os.path.expanduser(a.out)
    Dataset.from_list(rows).save_to_disk(out)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
