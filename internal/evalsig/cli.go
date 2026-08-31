// Command line interface, byte-compatible with the Python reference:
//
//	evalsig plan --baseline 0.42 --delta 0.03
//	evalsig report runs.json
//	evalsig check runs.json --factor seed
//	evalsig compare a.json b.json
//	evalsig decide candidates.json
//	evalsig seq runs.json
//
// Exit codes: 0 ok, 1 runtime failure (Python's uncaught exception path),
// 2 usage error (argparse convention) — and compare returns 2 when the
// verdict is INCONCLUSIVE, matching the reference CLI.
package evalsig

import (
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

type cliOpts struct {
	baseline     float64
	delta        float64
	alpha        float64
	power        float64
	discordance  float64
	width        float64
	widthWarning float64
	halfWidth    float64
	factors      []string
	nameA        string
	nameB        string
	jsonOut      bool
	runs         string
	a, b         string
	candidates   string
}

// Main is the CLI entry: args without the program name, exit code returned.
func Main(args []string, stdout, stderr io.Writer) (code int) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(stderr, "evalsig: error: %v\n", r)
			code = 1
		}
	}()
	if len(args) == 0 {
		return topErr(stderr, "the following arguments are required: cmd")
	}
	cmd := args[0]
	rest := args[1:]
	var c cliOpts
	c.baseline, c.delta, c.alpha, c.power = 0.5, 0.03, 0.05, 0.8
	c.discordance, c.width, c.widthWarning, c.halfWidth = 0.3, 0.05, 0.05, 0.05
	c.nameA, c.nameB = "A", "B"

	switch cmd {
	case "-h", "--help":
		fmt.Fprint(stdout, topHelp)
		return 0
	case "plan":
		_, extras, stop, code := parseOpts(rest, stdout, stderr, &planSub, nil, []optSpec{
			{"baseline", optFloat, &c.baseline},
			{"delta", optFloat, &c.delta},
			{"alpha", optFloat, &c.alpha},
			{"power", optFloat, &c.power},
			{"discordance", optFloat, &c.discordance},
			{"width", optFloat, &c.width},
		})
		if stop {
			return code
		}
		if e := extrasErr(stderr, extras); e != 0 {
			return e
		}
		return cmdPlan(&c, stdout)
	case "report":
		pos, extras, stop, code := parseOpts(rest, stdout, stderr, &reportSub, []string{"runs"}, []optSpec{
			{"alpha", optFloat, &c.alpha},
			{"factor", optStrAppend, &c.factors},
			{"width-warning", optFloat, &c.widthWarning},
		})
		if stop {
			return code
		}
		if e := requireArgs(stderr, &reportSub, pos, []string{"runs"}, nil); e != 0 {
			return e
		}
		if e := extrasErr(stderr, extras); e != 0 {
			return e
		}
		c.runs = pos[0]
		return cmdReport(&c, stdout, stderr)
	case "check":
		pos, extras, stop, code := parseOpts(rest, stdout, stderr, &checkSub, []string{"runs"}, []optSpec{
			{"factor", optStrAppend, &c.factors},
		})
		if stop {
			return code
		}
		var reqOpts []string
		if len(c.factors) == 0 {
			reqOpts = []string{"--factor"}
		}
		if e := requireArgs(stderr, &checkSub, pos, []string{"runs"}, reqOpts); e != 0 {
			return e
		}
		if e := extrasErr(stderr, extras); e != 0 {
			return e
		}
		c.runs = pos[0]
		return cmdCheck(&c, stdout, stderr)
	case "compare":
		pos, extras, stop, code := parseOpts(rest, stdout, stderr, &compareSub, []string{"a", "b"}, []optSpec{
			{"name-a", optStr, &c.nameA},
			{"name-b", optStr, &c.nameB},
			{"alpha", optFloat, &c.alpha},
			{"json", optBool, &c.jsonOut},
		})
		if stop {
			return code
		}
		if e := requireArgs(stderr, &compareSub, pos, []string{"a", "b"}, nil); e != 0 {
			return e
		}
		if e := extrasErr(stderr, extras); e != 0 {
			return e
		}
		c.a, c.b = pos[0], pos[1]
		return cmdCompare(&c, stdout, stderr)
	case "decide":
		pos, extras, stop, code := parseOpts(rest, stdout, stderr, &decideSub, []string{"candidates"}, []optSpec{
			{"alpha", optFloat, &c.alpha},
			{"json", optBool, &c.jsonOut},
		})
		if stop {
			return code
		}
		if e := requireArgs(stderr, &decideSub, pos, []string{"candidates"}, nil); e != 0 {
			return e
		}
		if e := extrasErr(stderr, extras); e != 0 {
			return e
		}
		c.candidates = pos[0]
		return cmdDecide(&c, stdout, stderr)
	case "seq":
		pos, extras, stop, code := parseOpts(rest, stdout, stderr, &seqSub, []string{"runs"}, []optSpec{
			{"alpha", optFloat, &c.alpha},
			{"half-width", optFloat, &c.halfWidth},
		})
		if stop {
			return code
		}
		if e := requireArgs(stderr, &seqSub, pos, []string{"runs"}, nil); e != 0 {
			return e
		}
		if e := extrasErr(stderr, extras); e != 0 {
			return e
		}
		c.runs = pos[0]
		return cmdSeq(&c, stdout, stderr)
	default:
		return topErr(stderr, "argument cmd: invalid choice: '"+cmd+
			"' (choose from 'plan', 'report', 'check', 'compare', 'decide', 'seq')")
	}
}

const topUsage = "usage: evalsig [-h] {plan,report,check,compare,decide,seq} ..."

const topHelp = `usage: evalsig [-h] {plan,report,check,compare,decide,seq} ...

Error bars and honest decisions for LLM/agent evaluation.

positional arguments:
  {plan,report,check,compare,decide,seq}
    plan                how many reruns do I need?
    report              estimate + CI + noise audit for one run set
    check               variance attribution by factor
    compare             A/B comparison with CI and verdict
    decide              rank candidates, eliminate the significantly worse
                        (Holm)
    seq                 replay runs through the sequential stopping gate

options:
  -h, --help            show this help message and exit
`

// subSpec carries a subcommand's argparse usage block (printed on usage
// errors) and full help text (printed for -h).
type subSpec struct {
	name  string
	usage string
	help  string
}

var planSub = subSpec{"plan",
	`usage: evalsig plan [-h] [--baseline BASELINE] [--delta DELTA] [--alpha ALPHA]
                    [--power POWER] [--discordance DISCORDANCE]
                    [--width WIDTH]
`,
	`usage: evalsig plan [-h] [--baseline BASELINE] [--delta DELTA] [--alpha ALPHA]
                    [--power POWER] [--discordance DISCORDANCE]
                    [--width WIDTH]

options:
  -h, --help            show this help message and exit
  --baseline BASELINE   expected success rate of the baseline
  --delta DELTA         effect size you must detect
  --alpha ALPHA
  --power POWER
  --discordance DISCORDANCE
                        expected task flip rate for paired design
  --width WIDTH         target CI width
`}

var reportSub = subSpec{"report",
	`usage: evalsig report [-h] [--alpha ALPHA] [--factor FACTOR]
                      [--width-warning WIDTH_WARNING]
                      runs
`,
	`usage: evalsig report [-h] [--alpha ALPHA] [--factor FACTOR]
                      [--width-warning WIDTH_WARNING]
                      runs

positional arguments:
  runs                  runs JSON file

options:
  -h, --help            show this help message and exit
  --alpha ALPHA
  --factor FACTOR       noise factor to attribute (repeatable)
  --width-warning WIDTH_WARNING
`}

var checkSub = subSpec{"check",
	"usage: evalsig check [-h] --factor FACTOR runs\n",
	`usage: evalsig check [-h] --factor FACTOR runs

positional arguments:
  runs

options:
  -h, --help       show this help message and exit
  --factor FACTOR
`}

var compareSub = subSpec{"compare",
	`usage: evalsig compare [-h] [--name-a NAME_A] [--name-b NAME_B]
                       [--alpha ALPHA] [--json]
                       a b
`,
	`usage: evalsig compare [-h] [--name-a NAME_A] [--name-b NAME_B]
                       [--alpha ALPHA] [--json]
                       a b

positional arguments:
  a
  b

options:
  -h, --help       show this help message and exit
  --name-a NAME_A
  --name-b NAME_B
  --alpha ALPHA
  --json
`}

var decideSub = subSpec{"decide",
	"usage: evalsig decide [-h] [--alpha ALPHA] [--json] candidates\n",
	`usage: evalsig decide [-h] [--alpha ALPHA] [--json] candidates

positional arguments:
  candidates     JSON: {name: [runs...]} or runs with a version field

options:
  -h, --help     show this help message and exit
  --alpha ALPHA
  --json
`}

var seqSub = subSpec{"seq",
	"usage: evalsig seq [-h] [--alpha ALPHA] [--half-width HALF_WIDTH] runs\n",
	`usage: evalsig seq [-h] [--alpha ALPHA] [--half-width HALF_WIDTH] runs

positional arguments:
  runs

options:
  -h, --help            show this help message and exit
  --alpha ALPHA
  --half-width HALF_WIDTH
`}

// topErr prints argparse's top-level usage error and returns the exit code.
func topErr(stderr io.Writer, msg string) int {
	fmt.Fprintf(stderr, "%s\nevalsig: error: %s\n", topUsage, msg)
	return 2
}

// subErr prints argparse's subcommand usage error and returns the exit code.
func subErr(stderr io.Writer, sub *subSpec, msg string) int {
	fmt.Fprintf(stderr, "%sevalsig %s: error: %s\n", sub.usage, sub.name, msg)
	return 2
}

// requireArgs reports argparse's missing-argument error: positionals in
// declaration order, then required options.
func requireArgs(stderr io.Writer, sub *subSpec, pos, posNames, reqOpts []string) int {
	var missing []string
	if len(pos) < len(posNames) {
		missing = append(missing, posNames[len(pos):]...)
	}
	missing = append(missing, reqOpts...)
	if len(missing) > 0 {
		return subErr(stderr, sub, "the following arguments are required: "+strings.Join(missing, ", "))
	}
	return 0
}

// extrasErr reports unrecognized arguments with the top-level usage, which
// is where argparse surfaces extras collected by a subparser.
func extrasErr(stderr io.Writer, extras []string) int {
	if len(extras) > 0 {
		return topErr(stderr, "unrecognized arguments: "+strings.Join(extras, " "))
	}
	return 0
}

type optKind int

const (
	optFloat optKind = iota
	optStr
	optBool
	optStrAppend
)

type optSpec struct {
	name string
	kind optKind
	dst  interface{}
}

// parseOpts is an argparse-compatible parser: exact names or unambiguous
// prefixes, --opt=value and --opt value forms, positionals in order.
// Returns positionals, unrecognized extras, and (stop, code): stop means
// help or a usage error was printed and the caller must return code.
func parseOpts(args []string, stdout, stderr io.Writer, sub *subSpec, posNames []string, specs []optSpec) (pos, extras []string, stop bool, code int) {
	takePos := func(v string) {
		if len(pos) < len(posNames) {
			pos = append(pos, v)
		} else {
			extras = append(extras, v)
		}
	}
	i := 0
	for i < len(args) {
		a := args[i]
		i++
		if a == "-h" || a == "--help" {
			fmt.Fprint(stdout, sub.help)
			return nil, nil, true, 0
		}
		if a == "--" {
			for i < len(args) {
				takePos(args[i])
				i++
			}
			break
		}
		if !strings.HasPrefix(a, "--") {
			takePos(a)
			continue
		}
		name := a
		val := ""
		hasVal := false
		if eq := strings.Index(a, "="); eq >= 0 {
			name, val, hasVal = a[:eq], a[eq+1:], true
		}
		spec, ok := matchOpt(name, specs)
		if !ok {
			extras = append(extras, a)
			continue
		}
		if spec.kind == optBool {
			if hasVal {
				subErr(stderr, sub, "argument --"+spec.name+": ignored explicit argument '"+val+"'")
				return nil, nil, true, 2
			}
			*(spec.dst.(*bool)) = true
			continue
		}
		if !hasVal {
			if i >= len(args) {
				subErr(stderr, sub, "argument --"+spec.name+": expected one argument")
				return nil, nil, true, 2
			}
			val = args[i]
			i++
		}
		switch spec.kind {
		case optFloat:
			v, err := parseFloatPy(val)
			if err != nil {
				subErr(stderr, sub, "argument --"+spec.name+": invalid float value: '"+val+"'")
				return nil, nil, true, 2
			}
			*spec.dst.(*float64) = v
		case optStr:
			*spec.dst.(*string) = val
		case optStrAppend:
			*spec.dst.(*[]string) = append(*spec.dst.(*[]string), val)
		}
	}
	return pos, extras, false, 0
}

// matchOpt resolves an option name with argparse's unique-prefix rule.
func matchOpt(name string, specs []optSpec) (optSpec, bool) {
	name = strings.TrimPrefix(name, "--")
	var found optSpec
	n := 0
	for _, s := range specs {
		if s.name == name {
			return s, true
		}
		if strings.HasPrefix(s.name, name) {
			found = s
			n++
		}
	}
	if n == 1 {
		return found, true
	}
	return optSpec{}, false
}

func parseFloatPy(s string) (float64, error) {
	return strconv.ParseFloat(strings.TrimSpace(s), 64)
}

func cmdPlan(c *cliOpts, w io.Writer) int {
	fmt.Fprintf(w, "planning to detect delta=%s at alpha=%s, power=%s\n",
		pyFormatF(c.delta, 3), pyReprFloat(c.alpha), pyReprFloat(c.power))
	if c.baseline >= 0 && c.baseline <= 1 {
		un := NTwoProportions(c.baseline, c.baseline+c.delta, c.alpha, c.power)
		fmt.Fprintf(w, "  unpaired (two independent run sets): %d runs per version\n", un.NPerVersion)
	}
	if c.discordance != 0 {
		paired := NPaired(c.discordance, c.delta, c.alpha, c.power)
		fmt.Fprintf(w, "  paired   (same task set, discordance=%s): %d tasks total\n",
			pyReprFloat(c.discordance), paired.NPerVersion)
		fmt.Fprintln(w, "  -> reusing one task set is typically 3-10x cheaper; always pair when you can")
	}
	p := c.baseline
	if !(c.baseline >= 0 && c.baseline <= 1) {
		p = 0.5
	}
	ci := NForCIWidth(c.width, p, c.alpha)
	fmt.Fprintf(w, "  to get CI width %s at p~%s: %d runs\n",
		pyFormatF(c.width, 2), pyReprFloat(p), ci.NPerVersion)
	return 0
}

func fmtCI(ci Interval) string {
	return fmt.Sprintf("[%s, %s] width=%s",
		fmt.Sprintf("%+.4f", ci.Low), fmt.Sprintf("%+.4f", ci.High), pyFormatF(ci.Width(), 4))
}

func cmdReport(c *cliOpts, stdout, stderr io.Writer) int {
	runs, err := LoadRuns(c.runs)
	if err != nil {
		return runErr(stderr, err)
	}
	if len(runs) == 0 {
		fmt.Fprintln(stdout, "no runs")
		return 1
	}
	binary := isBinary(runs)
	n := len(runs)
	outs := make([]float64, 0, n)
	for _, r := range runs {
		outs = append(outs, r.Outcome)
	}
	est := exactMean(outs)
	var ci *Interval
	if binary {
		s := 0.0
		for _, v := range outs {
			s += v
		}
		w := WilsonInterval(int(pyRound(s)), n, c.alpha)
		ci = &w
		fmt.Fprintf(stdout, "runs=%d  p_hat=%s  CI=%s  (wilson, %s)\n",
			n, pyFormatF(est, 4), fmtCI(w), pyPercent0(1-c.alpha))
	} else {
		fmt.Fprintf(stdout, "runs=%d  mean=%s  (continuous outcome; use compare for CIs)\n",
			n, pyFormatF(est, 4))
	}
	if c.widthWarning != 0 && ci != nil && ci.Width() > c.widthWarning {
		fmt.Fprintf(stdout, "  WARNING: CI width %s exceeds %s; run-to-run noise for agent benchmarks is commonly 2-6pp — your current n cannot separate changes of that size\n",
			pyFormatF(ci.Width(), 3), pyFormatF(c.widthWarning, 3))
	}
	for _, f := range c.factors {
		r := VarianceAttribution(runs, f)
		fmt.Fprintf(stdout, "\nvariance attribution by '%s': %s\n", f, r.Components.Summary())
		for _, g := range r.GroupMeans {
			fmt.Fprintf(stdout, "    %-28s mean=%+.4f\n", g.Key, g.Mean)
		}
		if r.Components.ICC > 0.3 {
			fmt.Fprintf(stdout, "    -> ICC %s: this factor explains a large share of noise; pin or stratify it\n",
				pyFormatF(r.Components.ICC, 2))
		}
	}
	return 0
}

func cmdCheck(c *cliOpts, stdout, stderr io.Writer) int {
	runs, err := LoadRuns(c.runs)
	if err != nil {
		return runErr(stderr, err)
	}
	for _, f := range c.factors {
		r := VarianceAttribution(runs, f)
		fmt.Fprintf(stdout, "factor '%s': %s\n", f, r.Components.Summary())
		for _, g := range r.GroupMeans {
			fmt.Fprintf(stdout, "    %-28s mean=%+.4f\n", g.Key, g.Mean)
		}
	}
	return 0
}

func cmdCompare(c *cliOpts, stdout, stderr io.Writer) int {
	a, err := LoadRuns(c.a)
	if err != nil {
		return runErr(stderr, err)
	}
	b, err := LoadRuns(c.b)
	if err != nil {
		return runErr(stderr, err)
	}
	comp, err := Compare(a, b, c.nameA, c.nameB, c.alpha)
	if err != nil {
		return runErr(stderr, err)
	}
	fmt.Fprintln(stdout, comp.String())
	if c.jsonOut {
		pv := interface{}(comp.PValue)
		if math.IsNaN(comp.PValue) {
			pv = nil
		}
		obj := newPyObj().
			set("estimate", comp.Estimate).
			set("ci", newPyObj().
				set("low", comp.CI.Low).
				set("high", comp.CI.High).
				set("width", comp.CI.Width())).
			set("p_value", pv).
			set("method", comp.Method).
			set("paired", comp.Paired).
			set("verdict", comp.Verdict).
			set("additional_needed", comp.AdditionalNeeded)
		fmt.Fprintln(stdout, pyJSONDumps(obj))
	}
	if comp.Verdict != "INCONCLUSIVE" {
		return 0
	}
	return 2
}

func cmdDecide(c *cliOpts, stdout, stderr io.Writer) int {
	data, err := os.ReadFile(c.candidates)
	if err != nil {
		return runErr(stderr, fmt.Errorf("%s: %v", c.candidates, err))
	}
	v, err := parsePyJSON(data)
	if err != nil {
		return runErr(stderr, fmt.Errorf("%s: %v", c.candidates, err))
	}
	var cands []Candidate
	if obj, ok := v.(*pyObj); ok {
		for _, kv := range obj.pairs {
			arr, ok := kv.v.([]interface{})
			if !ok {
				return runErr(stderr, fmt.Errorf("candidate %q is not a list of runs", kv.k))
			}
			runs, err := runsFromArray(arr, c.candidates)
			if err != nil {
				return runErr(stderr, err)
			}
			cands = append(cands, Candidate{kv.k, runs})
		}
	} else if arr, ok := v.([]interface{}); ok {
		seen := map[string]int{}
		for i, e := range arr {
			o, ok := e.(*pyObj)
			if !ok {
				return runErr(stderr, fmt.Errorf("%s: run %d is not a JSON object", c.candidates, i))
			}
			name := "?"
			if nv, ok := o.get("version"); ok {
				name = pyStr(nv)
			} else if nv, ok := o.get("name"); ok {
				name = pyStr(nv)
			}
			run, err := RunFromDict(o, i)
			if err != nil {
				return runErr(stderr, err)
			}
			if idx, ok := seen[name]; ok {
				cands[idx].Runs = append(cands[idx].Runs, run)
			} else {
				seen[name] = len(cands)
				cands = append(cands, Candidate{name, []Run{run}})
			}
		}
	} else {
		return runErr(stderr, fmt.Errorf("%s: expected a JSON object or array", c.candidates))
	}
	report, err := Decide(cands, c.alpha)
	if err != nil {
		return runErr(stderr, err)
	}
	fmt.Fprintln(stdout, report.String())
	if c.jsonOut {
		ranking := make([]interface{}, 0, len(report.Ranking))
		for _, e := range report.Ranking {
			ranking = append(ranking, newPyObj().
				set("name", e.Name).
				set("estimate", e.Estimate).
				set("action", e.Action))
		}
		adj := newPyObj()
		for _, h := range report.Adjusted {
			adj.set(h.Name, h.Adjusted)
		}
		fmt.Fprintln(stdout, pyJSONDumps(newPyObj().
			set("ranking", ranking).
			set("adjusted_p", adj)))
	}
	return 0
}

func runsFromArray(arr []interface{}, src string) ([]Run, error) {
	runs := make([]Run, 0, len(arr))
	for i, e := range arr {
		o, ok := e.(*pyObj)
		if !ok {
			return nil, fmt.Errorf("%s: run %d is not a JSON object", src, i)
		}
		r, err := RunFromDict(o, i)
		if err != nil {
			return nil, err
		}
		runs = append(runs, r)
	}
	return runs, nil
}

func cmdSeq(c *cliOpts, stdout, stderr io.Writer) int {
	runs, err := LoadRuns(c.runs)
	if err != nil {
		return runErr(stderr, err)
	}
	if !isBinary(runs) {
		fmt.Fprintln(stdout, "sequential gate currently supports binary outcomes; use compare for continuous")
		return 0
	}
	seq := NewSequentialProportion(c.alpha, c.halfWidth)
	s := 0
	for i, r := range runs {
		s += int(r.Outcome)
		v := seq.Update(s, i+1)
		if v != "CONTINUE" {
			fmt.Fprintf(stdout, "stopped at n=%d/%d: %s\n", i+1, len(runs), v)
			fmt.Fprintln(stdout, pyJSONDumps(seq.Report()))
			return 0
		}
	}
	fmt.Fprintf(stdout, "budget path: still CONTINUE at n=%d\n", len(runs))
	fmt.Fprintln(stdout, pyJSONDumps(seq.Report()))
	return 0
}

func runErr(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "evalsig: error: %v\n", err)
	return 1
}
