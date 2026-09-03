// cli.cpp — port of cli.go: argparse-compatible parsing and the six
// subcommands, byte-compatible with the Python reference CLI.

#include "cli.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "decide.h"
#include "intervals.h"
#include "power.h"
#include "pyjson.h"
#include "pynum.h"
#include "sequential.h"
#include "variance.h"

namespace evalsig {
namespace {

struct CliOpts {
    double baseline = 0.5;
    double delta = 0.03;
    double alpha = 0.05;
    double power = 0.8;
    double discordance = 0.3;
    double width = 0.05;
    double widthWarning = 0.05;
    double halfWidth = 0.05;
    std::vector<std::string> factors;
    std::string nameA = "A";
    std::string nameB = "B";
    bool jsonOut = false;
    std::string runs;
    std::string a;
    std::string b;
    std::string candidates;
};

const char* const kTopUsage = "usage: evalsig [-h] {plan,report,check,compare,decide,seq} ...";

const char* const kTopHelp =
    "usage: evalsig [-h] {plan,report,check,compare,decide,seq} ...\n"
    "\n"
    "Error bars and honest decisions for LLM/agent evaluation.\n"
    "\n"
    "positional arguments:\n"
    "  {plan,report,check,compare,decide,seq}\n"
    "    plan                how many reruns do I need?\n"
    "    report              estimate + CI + noise audit for one run set\n"
    "    check               variance attribution by factor\n"
    "    compare             A/B comparison with CI and verdict\n"
    "    decide              rank candidates, eliminate the significantly worse\n"
    "                        (Holm)\n"
    "    seq                 replay runs through the sequential stopping gate\n"
    "\n"
    "options:\n"
    "  -h, --help            show this help message and exit\n";

// A subcommand's argparse usage block (printed on usage errors) and full
// help text (printed for -h).
struct SubSpec {
    const char* name = "";
    const char* usage = "";
    const char* help = "";
};

const SubSpec kPlanSub{"plan",
                       "usage: evalsig plan [-h] [--baseline BASELINE] [--delta DELTA] "
                       "[--alpha ALPHA]\n"
                       "                    [--power POWER] [--discordance DISCORDANCE]\n"
                       "                    [--width WIDTH]\n",
                       "usage: evalsig plan [-h] [--baseline BASELINE] [--delta DELTA] "
                       "[--alpha ALPHA]\n"
                       "                    [--power POWER] [--discordance DISCORDANCE]\n"
                       "                    [--width WIDTH]\n"
                       "\n"
                       "options:\n"
                       "  -h, --help            show this help message and exit\n"
                       "  --baseline BASELINE   expected success rate of the baseline\n"
                       "  --delta DELTA         effect size you must detect\n"
                       "  --alpha ALPHA\n"
                       "  --power POWER\n"
                       "  --discordance DISCORDANCE\n"
                       "                        expected task flip rate for paired design\n"
                       "  --width WIDTH         target CI width\n"};

const SubSpec kReportSub{"report",
                         "usage: evalsig report [-h] [--alpha ALPHA] [--factor FACTOR]\n"
                         "                      [--width-warning WIDTH_WARNING]\n"
                         "                      runs\n",
                         "usage: evalsig report [-h] [--alpha ALPHA] [--factor FACTOR]\n"
                         "                      [--width-warning WIDTH_WARNING]\n"
                         "                      runs\n"
                         "\n"
                         "positional arguments:\n"
                         "  runs                  runs JSON file\n"
                         "\n"
                         "options:\n"
                         "  -h, --help            show this help message and exit\n"
                         "  --alpha ALPHA\n"
                         "  --factor FACTOR       noise factor to attribute (repeatable)\n"
                         "  --width-warning WIDTH_WARNING\n"};

const SubSpec kCheckSub{"check",
                        "usage: evalsig check [-h] --factor FACTOR runs\n",
                        "usage: evalsig check [-h] --factor FACTOR runs\n"
                        "\n"
                        "positional arguments:\n"
                        "  runs\n"
                        "\n"
                        "options:\n"
                        "  -h, --help       show this help message and exit\n"
                        "  --factor FACTOR\n"};

const SubSpec kCompareSub{"compare",
                          "usage: evalsig compare [-h] [--name-a NAME_A] [--name-b NAME_B]\n"
                          "                       [--alpha ALPHA] [--json]\n"
                          "                       a b\n",
                          "usage: evalsig compare [-h] [--name-a NAME_A] [--name-b NAME_B]\n"
                          "                       [--alpha ALPHA] [--json]\n"
                          "                       a b\n"
                          "\n"
                          "positional arguments:\n"
                          "  a\n"
                          "  b\n"
                          "\n"
                          "options:\n"
                          "  -h, --help       show this help message and exit\n"
                          "  --name-a NAME_A\n"
                          "  --name-b NAME_B\n"
                          "  --alpha ALPHA\n"
                          "  --json\n"};

const SubSpec kDecideSub{"decide",
                         "usage: evalsig decide [-h] [--alpha ALPHA] [--json] candidates\n",
                         "usage: evalsig decide [-h] [--alpha ALPHA] [--json] candidates\n"
                         "\n"
                         "positional arguments:\n"
                         "  candidates     JSON: {name: [runs...]} or runs with a version field\n"
                         "\n"
                         "options:\n"
                         "  -h, --help     show this help message and exit\n"
                         "  --alpha ALPHA\n"
                         "  --json\n"};

const SubSpec kSeqSub{"seq",
                      "usage: evalsig seq [-h] [--alpha ALPHA] [--half-width HALF_WIDTH] runs\n",
                      "usage: evalsig seq [-h] [--alpha ALPHA] [--half-width HALF_WIDTH] runs\n"
                      "\n"
                      "positional arguments:\n"
                      "  runs\n"
                      "\n"
                      "options:\n"
                      "  -h, --help            show this help message and exit\n"
                      "  --alpha ALPHA\n"
                      "  --half-width HALF_WIDTH\n"};

// argparse's top-level usage error.
int topErr(std::ostream& err, const std::string& msg) {
    err << kTopUsage << "\nevalsig: error: " << msg << "\n";
    return 2;
}

// argparse's subcommand usage error.
int subErr(std::ostream& err, const SubSpec& sub, const std::string& msg) {
    err << sub.usage << "evalsig " << sub.name << ": error: " << msg << "\n";
    return 2;
}

// argparse's missing-argument error: positionals in declaration order, then
// required options.
int requireArgs(std::ostream& err, const SubSpec& sub,
                const std::vector<std::string>& pos,
                const std::vector<std::string>& posNames,
                const std::vector<std::string>& reqOpts) {
    std::vector<std::string> missing;
    if (pos.size() < posNames.size()) {
        missing.insert(missing.end(), posNames.begin() + static_cast<long>(pos.size()),
                      posNames.end());
    }
    missing.insert(missing.end(), reqOpts.begin(), reqOpts.end());
    if (!missing.empty()) {
        std::string joined = missing[0];
        for (size_t i = 1; i < missing.size(); i++) joined += ", " + missing[i];
        return subErr(err, sub, "the following arguments are required: " + joined);
    }
    return 0;
}

// Unrecognized arguments surface with the top-level usage, which is where
// argparse reports extras collected by a subparser.
int extrasErr(std::ostream& err, const std::vector<std::string>& extras) {
    if (!extras.empty()) {
        std::string joined = extras[0];
        for (size_t i = 1; i < extras.size(); i++) joined += " " + extras[i];
        return topErr(err, "unrecognized arguments: " + joined);
    }
    return 0;
}

enum class OptKind { Float, Str, Bool, StrAppend };

struct OptSpec {
    const char* name = "";
    OptKind kind = OptKind::Float;
    double* fdst = nullptr;
    std::string* sdst = nullptr;
    bool* bdst = nullptr;
    std::vector<std::string>* adst = nullptr;
};

bool parseFloatPy(const std::string& s, double& out) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\v' ||
                     s[b] == '\f' || s[b] == '\r')) {
        b++;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' ||
                     s[e - 1] == '\v' || s[e - 1] == '\f' || s[e - 1] == '\r')) {
        e--;
    }
    if (b == e) {
        return false;
    }
    std::string t = s.substr(b, e - b);
    const char* cs = t.c_str();
    char* end = nullptr;
    out = std::strtod(cs, &end);
    return end != nullptr && *end == '\0';
}

// Resolves an option name with argparse's unique-prefix rule.
bool matchOpt(const std::string& argName, const std::vector<OptSpec>& specs, OptSpec& out) {
    std::string name = argName;
    if (name.rfind("--", 0) == 0) {
        name = name.substr(2);
    }
    int n = 0;
    for (const auto& s : specs) {
        if (s.name == name) {
            out = s;
            return true;
        }
        if (std::string(s.name).rfind(name, 0) == 0) {
            out = s;
            n++;
        }
    }
    return n == 1;
}

struct ParseResult {
    std::vector<std::string> pos;
    std::vector<std::string> extras;
    bool stop = false;
    int code = 0;
};

// An argparse-compatible parser: exact names or unambiguous prefixes,
// --opt=value and --opt value forms, positionals in order.
ParseResult parseOpts(const std::vector<std::string>& args, std::ostream& out,
                      std::ostream& err, const SubSpec& sub,
                      const std::vector<std::string>& posNames,
                      const std::vector<OptSpec>& specs) {
    ParseResult r;
    auto takePos = [&](const std::string& v) {
        if (r.pos.size() < posNames.size()) {
            r.pos.push_back(v);
        } else {
            r.extras.push_back(v);
        }
    };
    size_t i = 0;
    while (i < args.size()) {
        std::string a = args[i];
        i++;
        if (a == "-h" || a == "--help") {
            out << sub.help;
            r.stop = true;
            r.code = 0;
            return r;
        }
        if (a == "--") {
            while (i < args.size()) {
                takePos(args[i]);
                i++;
            }
            break;
        }
        if (a.rfind("--", 0) != 0) {
            takePos(a);
            continue;
        }
        std::string name = a;
        std::string val;
        bool hasVal = false;
        size_t eq = a.find('=');
        if (eq != std::string::npos) {
            name = a.substr(0, eq);
            val = a.substr(eq + 1);
            hasVal = true;
        }
        OptSpec spec;
        if (!matchOpt(name, specs, spec)) {
            r.extras.push_back(a);
            continue;
        }
        if (spec.kind == OptKind::Bool) {
            if (hasVal) {
                subErr(err, sub, "argument --" + std::string(spec.name) +
                                        ": ignored explicit argument '" + val + "'");
                r.stop = true;
                r.code = 2;
                return r;
            }
            *spec.bdst = true;
            continue;
        }
        if (!hasVal) {
            if (i >= args.size()) {
                subErr(err, sub,
                       "argument --" + std::string(spec.name) + ": expected one argument");
                r.stop = true;
                r.code = 2;
                return r;
            }
            val = args[i];
            i++;
        }
        switch (spec.kind) {
            case OptKind::Float: {
                double v;
                if (!parseFloatPy(val, v)) {
                    subErr(err, sub, "argument --" + std::string(spec.name) +
                                            ": invalid float value: '" + val + "'");
                    r.stop = true;
                    r.code = 2;
                    return r;
                }
                *spec.fdst = v;
                break;
            }
            case OptKind::Str:
                *spec.sdst = val;
                break;
            case OptKind::StrAppend:
                spec.adst->push_back(val);
                break;
            default:
                break;
        }
    }
    return r;
}

std::string fmtCI(const Interval& ci) {
    return "[" + fmtPlusF(ci.low, 4) + ", " + fmtPlusF(ci.high, 4) + "] width=" +
           pyFormatF(ci.width(), 4);
}

int runErr(std::ostream& err, const std::string& msg) {
    err << "evalsig: error: " << msg << "\n";
    return 1;
}

int cmdPlan(const CliOpts& c, std::ostream& w) {
    w << "planning to detect delta=" << pyFormatF(c.delta, 3) << " at alpha="
      << pyReprFloat(c.alpha) << ", power=" << pyReprFloat(c.power) << "\n";
    if (c.baseline >= 0 && c.baseline <= 1) {
        PowerPlan un = nTwoProportions(c.baseline, c.baseline + c.delta, c.alpha, c.power);
        w << "  unpaired (two independent run sets): " << un.nPerVersion
          << " runs per version\n";
    }
    if (c.discordance != 0) {
        PowerPlan paired = nPaired(c.discordance, c.delta, c.alpha, c.power);
        w << "  paired   (same task set, discordance=" << pyReprFloat(c.discordance)
          << "): " << paired.nPerVersion << " tasks total\n";
        w << "  -> reusing one task set is typically 3-10x cheaper; always pair when you can\n";
    }
    double p = c.baseline;
    if (!(c.baseline >= 0 && c.baseline <= 1)) {
        p = 0.5;
    }
    PowerPlan ci = nForCIWidth(c.width, p, c.alpha);
    w << "  to get CI width " << pyFormatF(c.width, 2) << " at p~" << pyReprFloat(p) << ": "
      << ci.nPerVersion << " runs\n";
    return 0;
}

int cmdReport(const CliOpts& c, std::ostream& out, std::ostream& err) {
    std::vector<Run> runs;
    try {
        runs = loadRuns(c.runs);
    } catch (const std::exception& e) {
        return runErr(err, e.what());
    }
    if (runs.empty()) {
        out << "no runs\n";
        return 1;
    }
    bool binary = isBinary(runs);
    int n = static_cast<int>(runs.size());
    std::vector<double> outs;
    outs.reserve(runs.size());
    for (const Run& r : runs) {
        outs.push_back(r.outcome);
    }
    double est = exactMean(outs);
    Interval w = wilsonInterval(0, 1, c.alpha);
    bool hasCI = false;
    if (binary) {
        double s = 0.0;
        for (double v : outs) {
            s += v;
        }
        w = wilsonInterval(static_cast<int>(pyRound(s)), n, c.alpha);
        hasCI = true;
        out << "runs=" << n << "  p_hat=" << pyFormatF(est, 4) << "  CI=" << fmtCI(w)
               << "  (wilson, " << pyPercent0(1 - c.alpha) << ")\n";
    } else {
        out << "runs=" << n << "  mean=" << pyFormatF(est, 4)
               << "  (continuous outcome; use compare for CIs)\n";
    }
    if (c.widthWarning != 0 && hasCI && w.width() > c.widthWarning) {
        out << "  WARNING: CI width " << pyFormatF(w.width(), 3) << " exceeds "
               << pyFormatF(c.widthWarning, 3)
               << "; run-to-run noise for agent benchmarks is commonly 2-6pp — "
                  "your current n cannot separate changes of that size\n";
    }
    for (const auto& f : c.factors) {
        Attribution r = varianceAttribution(runs, f);
        out << "\nvariance attribution by '" << f
               << "': " << r.components.summary() << "\n";
        for (const auto& g : r.groupMeans) {
            out << "    " << fmtPadLeft(g.key, 28) << " mean=" << fmtPlusF(g.mean, 4)
                   << "\n";
        }
        if (r.components.icc > 0.3) {
            out << "    -> ICC " << pyFormatF(r.components.icc, 2)
                  << ": this factor explains a large share of noise; pin or stratify it\n";
        }
    }
    return 0;
}

int cmdCheck(const CliOpts& c, std::ostream& out, std::ostream& err) {
    std::vector<Run> runs;
    try {
        runs = loadRuns(c.runs);
    } catch (const std::exception& e) {
        return runErr(err, e.what());
    }
    for (const auto& f : c.factors) {
        Attribution r = varianceAttribution(runs, f);
        out << "factor '" << f << "': " << r.components.summary() << "\n";
        for (const auto& g : r.groupMeans) {
            out << "    " << fmtPadLeft(g.key, 28) << " mean=" << fmtPlusF(g.mean, 4)
                   << "\n";
        }
    }
    return 0;
}

int cmdCompare(const CliOpts& c, std::ostream& out, std::ostream& err) {
    std::vector<Run> a;
    std::vector<Run> b;
    try {
        a = loadRuns(c.a);
        b = loadRuns(c.b);
    } catch (const std::exception& e) {
        return runErr(err, e.what());
    }
    Comparison comp = compare(a, b, c.nameA, c.nameB, c.alpha);
    out << comp.str() << "\n";
    if (c.jsonOut) {
        PyValue pv;
        if (std::isnan(comp.pValue)) {
            pv = PyValue{};  // null
        } else {
            pv = comp.pValue;
        }
        auto ciObj = std::make_shared<PyObj>();
        ciObj->set("low", comp.ci.low).set("high", comp.ci.high).set("width", comp.ci.width());
        auto obj = std::make_shared<PyObj>();
        obj->set("estimate", comp.estimate)
            .set("ci", ciObj)
            .set("p_value", pv)
            .set("method", comp.method)
            .set("paired", comp.paired)
            .set("verdict", comp.verdict)
            .set("additional_needed", static_cast<long long>(comp.additionalNeeded));
        out << pyJSONDumps(obj) << "\n";
    }
    if (comp.verdict != "INCONCLUSIVE") {
        return 0;
    }
    return 2;
}

int cmdDecide(const CliOpts& c, std::ostream& out, std::ostream& err) {
    std::string data;
    try {
        data = readFileOrThrow(c.candidates);
    } catch (const std::exception& e) {
        return runErr(err, c.candidates + ": " + e.what());
    }
    PyValue v;
    try {
        v = parsePyJSON(data);
    } catch (const std::exception& e) {
        return runErr(err, c.candidates + ": " + e.what());
    }
    std::vector<Candidate> cands;
    if (const auto* obj = std::get_if<std::shared_ptr<PyObj>>(&v.v)) {
        for (const auto& kv : (*obj)->pairs) {
            const auto* arr = std::get_if<std::vector<PyValue>>(&kv.second.v);
            if (!arr) {
                return runErr(err,
                              "candidate " + goQuote(kv.first) + " is not a list of runs");
            }
            std::vector<Run> runs;
            try {
                runs = runsFromArray(*arr, c.candidates);
            } catch (const std::exception& e) {
                return runErr(err, e.what());
            }
            Candidate cd;
            cd.name = kv.first;
            cd.runs = std::move(runs);
            cands.push_back(std::move(cd));
        }
    } else if (const auto* arr = std::get_if<std::vector<PyValue>>(&v.v)) {
        std::map<std::string, size_t> seen;
        for (size_t i = 0; i < arr->size(); i++) {
            const auto* o = std::get_if<std::shared_ptr<PyObj>>(&(*arr)[i].v);
            if (!o) {
                return runErr(err, c.candidates + ": run " + std::to_string(i) +
                                          " is not a JSON object");
            }
            std::string name = "?";
            if (const PyValue* nv = (*o)->get("version")) {
                name = pyStr(*nv);
            } else if (const PyValue* nv = (*o)->get("name")) {
                name = pyStr(*nv);
            }
            Run run;
            try {
                run = runFromDict(**o, static_cast<int>(i));
            } catch (const std::exception& e) {
                return runErr(err, e.what());
            }
            auto it = seen.find(name);
            if (it != seen.end()) {
                cands[it->second].runs.push_back(std::move(run));
            } else {
                seen[name] = cands.size();
                Candidate cd;
                cd.name = name;
                cd.runs.push_back(std::move(run));
                cands.push_back(std::move(cd));
            }
        }
    } else {
        return runErr(err, c.candidates + ": expected a JSON object or array");
    }
    DecisionReport report;
    try {
        report = decide(cands, c.alpha);
    } catch (const std::exception& e) {
        return runErr(err, e.what());
    }
    out << report.str() << "\n";
    if (c.jsonOut) {
        std::vector<PyValue> ranking;
        ranking.reserve(report.ranking.size());
        for (const auto& e : report.ranking) {
            auto o = std::make_shared<PyObj>();
            o->set("name", e.name).set("estimate", e.estimate).set("action", e.action);
            ranking.push_back(o);
        }
        auto adj = std::make_shared<PyObj>();
        for (const auto& h : report.adjusted) {
            adj->set(h.name, h.adjusted);
        }
        auto obj = std::make_shared<PyObj>();
        obj->set("ranking", ranking).set("adjusted_p", adj);
        out << pyJSONDumps(obj) << "\n";
    }
    return 0;
}

int cmdSeq(const CliOpts& c, std::ostream& out, std::ostream& err) {
    std::vector<Run> runs;
    try {
        runs = loadRuns(c.runs);
    } catch (const std::exception& e) {
        return runErr(err, e.what());
    }
    if (!isBinary(runs)) {
        out << "sequential gate currently supports binary outcomes; use compare for "
                  "continuous\n";
        return 0;
    }
    SequentialProportion seq(c.alpha, c.halfWidth);
    int s = 0;
    for (size_t i = 0; i < runs.size(); i++) {
        s += static_cast<int>(runs[i].outcome);
        std::string v = seq.update(s, static_cast<int>(i) + 1);
        if (v != "CONTINUE") {
            out << "stopped at n=" << (i + 1) << "/" << runs.size() << ": " << v << "\n";
            out << pyJSONDumps(seq.report()) << "\n";
            return 0;
        }
    }
    out << "budget path: still CONTINUE at n=" << runs.size() << "\n";
    out << pyJSONDumps(seq.report()) << "\n";
    return 0;
}

}  // namespace

int cliMain(const std::vector<std::string>& args, std::ostream& out,
            std::ostream& err) {
    try {
        if (args.empty()) {
            return topErr(err, "the following arguments are required: cmd");
        }
        std::string cmd = args[0];
        std::vector<std::string> rest(args.begin() + 1, args.end());
        CliOpts c;
        if (cmd == "-h" || cmd == "--help") {
            out << kTopHelp;
            return 0;
        }
        if (cmd == "plan") {
            ParseResult r = parseOpts(rest, out, err, kPlanSub, {},
                                      {OptSpec{"baseline", OptKind::Float, &c.baseline},
                                       OptSpec{"delta", OptKind::Float, &c.delta},
                                       OptSpec{"alpha", OptKind::Float, &c.alpha},
                                       OptSpec{"power", OptKind::Float, &c.power},
                                       OptSpec{"discordance", OptKind::Float, &c.discordance},
                                       OptSpec{"width", OptKind::Float, &c.width}});
            if (r.stop) return r.code;
            int e = extrasErr(err, r.extras);
            if (e != 0) return e;
            return cmdPlan(c, out);
        }
        if (cmd == "report") {
            ParseResult r =
                parseOpts(rest, out, err, kReportSub, {"runs"},
                          {OptSpec{"alpha", OptKind::Float, &c.alpha},
                           OptSpec{"factor", OptKind::StrAppend, nullptr, nullptr, nullptr, &c.factors},
                           OptSpec{"width-warning", OptKind::Float, &c.widthWarning}});
            if (r.stop) return r.code;
            int e = requireArgs(err, kReportSub, r.pos, {"runs"}, {});
            if (e != 0) return e;
            e = extrasErr(err, r.extras);
            if (e != 0) return e;
            c.runs = r.pos[0];
            return cmdReport(c, out, err);
        }
        if (cmd == "check") {
            ParseResult r =
                parseOpts(rest, out, err, kCheckSub, {"runs"},
                          {OptSpec{"factor", OptKind::StrAppend, nullptr, nullptr, nullptr, &c.factors}});
            if (r.stop) return r.code;
            std::vector<std::string> reqOpts;
            if (c.factors.empty()) {
                reqOpts = {"--factor"};
            }
            int e = requireArgs(err, kCheckSub, r.pos, {"runs"}, reqOpts);
            if (e != 0) return e;
            e = extrasErr(err, r.extras);
            if (e != 0) return e;
            c.runs = r.pos[0];
            return cmdCheck(c, out, err);
        }
        if (cmd == "compare") {
            ParseResult r =
                parseOpts(rest, out, err, kCompareSub, {"a", "b"},
                          {OptSpec{"name-a", OptKind::Str, nullptr, &c.nameA},
                           OptSpec{"name-b", OptKind::Str, nullptr, &c.nameB},
                           OptSpec{"alpha", OptKind::Float, &c.alpha},
                           OptSpec{"json", OptKind::Bool, nullptr, nullptr, &c.jsonOut}});
            if (r.stop) return r.code;
            int e = requireArgs(err, kCompareSub, r.pos, {"a", "b"}, {});
            if (e != 0) return e;
            e = extrasErr(err, r.extras);
            if (e != 0) return e;
            c.a = r.pos[0];
            c.b = r.pos[1];
            return cmdCompare(c, out, err);
        }
        if (cmd == "decide") {
            ParseResult r =
                parseOpts(rest, out, err, kDecideSub, {"candidates"},
                          {OptSpec{"alpha", OptKind::Float, &c.alpha},
                           OptSpec{"json", OptKind::Bool, nullptr, nullptr, &c.jsonOut}});
            if (r.stop) return r.code;
            int e = requireArgs(err, kDecideSub, r.pos, {"candidates"}, {});
            if (e != 0) return e;
            e = extrasErr(err, r.extras);
            if (e != 0) return e;
            c.candidates = r.pos[0];
            return cmdDecide(c, out, err);
        }
        if (cmd == "seq") {
            ParseResult r =
                parseOpts(rest, out, err, kSeqSub, {"runs"},
                          {OptSpec{"alpha", OptKind::Float, &c.alpha},
                           OptSpec{"half-width", OptKind::Float, &c.halfWidth}});
            if (r.stop) return r.code;
            int e = requireArgs(err, kSeqSub, r.pos, {"runs"}, {});
            if (e != 0) return e;
            e = extrasErr(err, r.extras);
            if (e != 0) return e;
            c.runs = r.pos[0];
            return cmdSeq(c, out, err);
        }
        return topErr(err, "argument cmd: invalid choice: '" + cmd +
                                  "' (choose from 'plan', 'report', 'check', 'compare', "
                                  "'decide', 'seq')");
    } catch (const std::exception& e) {
        err << "evalsig: error: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace evalsig
