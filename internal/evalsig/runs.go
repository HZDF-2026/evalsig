// Run records and the JSON schema they load from, with CPython-faithful
// semantics: object key order is preserved (insertion order drives dict
// iteration in variance attribution and decide ranking ties), and numbers
// keep their int/float distinction (str(3) == "3" but str(3.0) == "3.0").
package evalsig

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

// pyValue is a parsed JSON value: nil, bool, string, json.Number,
// []interface{} or *pyObj.
type pyValue = interface{}

type pyKV struct {
	k string
	v pyValue
}

// pyObj is an insertion-ordered object: it is both what parsePyJSON
// produces and what the JSON emitters build (Python dicts keep insertion
// order in both directions).
type pyObj struct {
	pairs []pyKV
}

func newPyObj() *pyObj { return &pyObj{} }

func (o *pyObj) set(k string, v pyValue) *pyObj {
	o.pairs = append(o.pairs, pyKV{k, v})
	return o
}

func (o *pyObj) get(key string) (pyValue, bool) {
	for _, kv := range o.pairs {
		if kv.k == key {
			return kv.v, true
		}
	}
	return nil, false
}

func (o *pyObj) keys() []string {
	ks := make([]string, 0, len(o.pairs))
	for _, kv := range o.pairs {
		ks = append(ks, kv.k)
	}
	return ks
}

// parsePyJSON parses data as JSON preserving object key order and the
// textual form of numbers (json.Number).
func parsePyJSON(data []byte) (pyValue, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := parsePyValue(dec)
	if err != nil {
		return nil, err
	}
	if tok, err := dec.Token(); err != io.EOF {
		_ = tok
		return nil, fmt.Errorf("extra data after JSON document")
	}
	return v, nil
}

func parsePyValue(dec *json.Decoder) (pyValue, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	return tokenToValue(dec, tok)
}

func tokenToValue(dec *json.Decoder, tok json.Token) (pyValue, error) {
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			o := &pyObj{}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, ok := kt.(string)
				if !ok {
					return nil, fmt.Errorf("object key is not a string")
				}
				v, err := parsePyValue(dec)
				if err != nil {
					return nil, err
				}
				o.pairs = append(o.pairs, pyKV{key, v})
			}
			if _, err := dec.Token(); err != nil { // closing }
				return nil, err
			}
			return o, nil
		case '[':
			arr := []interface{}{}
			for dec.More() {
				v, err := parsePyValue(dec)
				if err != nil {
					return nil, err
				}
				arr = append(arr, v)
			}
			if _, err := dec.Token(); err != nil { // closing ]
				return nil, err
			}
			return arr, nil
		}
		return nil, fmt.Errorf("unexpected delimiter %v", t)
	default:
		return tok, nil
	}
}

// numFloat converts a json.Number to float64 like Python float(x).
func numFloat(n json.Number) (float64, error) {
	return strconv.ParseFloat(string(n), 64)
}

// numIsInt reports whether the JSON number text denotes an int (no '.', 'e',
// 'E') — matching json.load's int/float split.
func numIsInt(n json.Number) bool {
	s := string(n)
	return !strings.ContainsAny(s, ".eE")
}

// pyStr is Python str(v) for parsed JSON values.
func pyStr(v pyValue) string {
	switch t := v.(type) {
	case nil:
		return "None"
	case bool:
		if t {
			return "True"
		}
		return "False"
	case string:
		return t
	case json.Number:
		if numIsInt(t) {
			return t.String()
		}
		f, err := numFloat(t)
		if err != nil {
			return t.String()
		}
		return pyReprFloat(f)
	default:
		return pyRepr(v)
	}
}

// pyRepr is Python repr(v) for the composite values.
func pyRepr(v pyValue) string {
	switch t := v.(type) {
	case nil:
		return "None"
	case bool:
		if t {
			return "True"
		}
		return "False"
	case string:
		return "'" + strings.ReplaceAll(t, "'", `\'`) + "'"
	case json.Number:
		return pyStr(t)
	case []interface{}:
		parts := make([]string, 0, len(t))
		for _, e := range t {
			parts = append(parts, pyRepr(e))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case *pyObj:
		parts := make([]string, 0, len(t.pairs))
		for _, kv := range t.pairs {
			parts = append(parts, pyRepr(kv.k)+": "+pyRepr(kv.v))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	}
	return fmt.Sprintf("%v", v)
}

// pyNumberToFloat converts a value like Python float(v): bool -> 0.0/1.0,
// number -> itself, string -> parsed.
func pyNumberToFloat(v pyValue) (float64, error) {
	switch t := v.(type) {
	case bool:
		if t {
			return 1.0, nil
		}
		return 0.0, nil
	case json.Number:
		return numFloat(t)
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(t), 64)
		if err != nil {
			return 0, fmt.Errorf("could not convert string to float: %q", t)
		}
		return f, nil
	}
	return 0, fmt.Errorf("must be a real number, not %s", pyTypeName(v))
}

func pyTypeName(v pyValue) string {
	switch v.(type) {
	case nil:
		return "NoneType"
	case bool:
		return "bool"
	case string:
		return "str"
	case json.Number:
		return "int or float"
	case []interface{}:
		return "list"
	case *pyObj:
		return "dict"
	}
	return "unknown"
}

// outcomeKeys is the Python _outcome probe order.
var outcomeKeys = []string{"success", "resolved", "passed", "score", "value", "outcome"}

// outcomeOf extracts the run outcome like Python _outcome(d).
func outcomeOf(o *pyObj) (float64, error) {
	for _, k := range outcomeKeys {
		if v, ok := o.get(k); ok {
			return pyNumberToFloat(v)
		}
	}
	return 0, &OutcomeError{Keys: o.keys()}
}

// Run is one evaluation run: task id, outcome, and all factor fields.
type Run struct {
	Task    string
	Outcome float64
	Order   int      // position in the file, tie-break for stable sorts
	Factors *pyObj   // all fields except task/id, in file order
}

// RunFromDict converts a parsed run object.
func RunFromDict(o *pyObj, order int) (Run, error) {
	task := ""
	if v, ok := o.get("task"); ok {
		task = pyStr(v)
	} else if v, ok := o.get("id"); ok {
		task = pyStr(v)
	}
	outcome, err := outcomeOf(o)
	if err != nil {
		return Run{}, err
	}
	f := &pyObj{}
	for _, kv := range o.pairs {
		if kv.k == "task" || kv.k == "id" {
			continue
		}
		f.pairs = append(f.pairs, kv)
	}
	return Run{Task: task, Outcome: outcome, Order: order, Factors: f}, nil
}

// factorValue returns the factor's raw value (Python r.get(factor)).
func (r Run) factorValue(factor string) (pyValue, bool) {
	if r.Factors == nil {
		return nil, false
	}
	return r.Factors.get(factor)
}

// FlatKeys returns run factors in file order (task/outcome are not factors).
func (r Run) FlatKeys() []string {
	if r.Factors == nil {
		return nil
	}
	return r.Factors.keys()
}

// LoadRuns reads a JSON array of run dicts; see README for the schema.
func LoadRuns(path string) ([]Run, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	v, err := parsePyJSON(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil, fmt.Errorf("%s: expected a JSON array of runs", path)
	}
	runs := make([]Run, 0, len(arr))
	for i, e := range arr {
		o, ok := e.(*pyObj)
		if !ok {
			return nil, fmt.Errorf("%s: run %d is not a JSON object", path, i)
		}
		r, err := RunFromDict(o, i)
		if err != nil {
			return nil, err
		}
		runs = append(runs, r)
	}
	return runs, nil
}

// sortedStringSet returns sorted unique keys (Python sorted(set(...))).
func sortedStringSet(keys []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		if !seen[k] {
			seen[k] = true
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}
