// pyjson.h — the CPython-faithful JSON data model: parsed values keep their
// object key order (insertion order drives dict iteration in variance
// attribution and decide ranking ties) and their int/float distinction
// (str(3) == "3" but str(3.0) == "3.0"); json.dumps(indent=2) reproduces
// Python's separators, ensure_ascii escaping, and float repr.
//
// Port of runs.go plus the JSON emitters of pystat.go.
#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace evalsig {

struct PyObj;

// A JSON number kept as its verbatim text (Go's json.Number): ints re-emit
// verbatim, floats re-emit through Python's repr.
struct PyNum {
    std::string text;
};

// Parsed JSON value: null (monostate), bool, string, verbatim number,
// int64/float64 (values built in memory, never parsed), array, or object.
// A struct (not a bare alias) so the array alternative can recurse.
struct PyValue {
    using Impl = std::variant<std::monostate, bool, std::string, PyNum, long long, double,
                              std::vector<PyValue>, std::shared_ptr<PyObj>>;
    Impl v;

    PyValue() : v(std::monostate{}) {}
    PyValue(const PyValue&) = default;
    PyValue(PyValue&&) = default;
    PyValue& operator=(const PyValue&) = default;
    PyValue& operator=(PyValue&&) = default;

    template <typename T,
              typename = std::enable_if_t<std::is_constructible_v<Impl, T&&> &&
                                          !std::is_same_v<std::decay_t<T>, PyValue>>>
    PyValue(T&& x) : v(std::forward<T>(x)) {}
};

// Insertion-ordered object.
struct PyObj {
    std::vector<std::pair<std::string, PyValue>> pairs;

    PyObj& set(const std::string& k, PyValue v) {
        pairs.emplace_back(k, std::move(v));
        return *this;
    }
    const PyValue* get(const std::string& key) const {
        for (const auto& kv : pairs) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
    std::vector<std::string> keys() const {
        std::vector<std::string> ks;
        ks.reserve(pairs.size());
        for (const auto& kv : pairs) ks.push_back(kv.first);
        return ks;
    }
};

// Parses data as JSON, preserving key order and number text. Throws
// std::runtime_error on invalid input or data after the document.
PyValue parsePyJSON(const std::string& data);

bool numIsInt(const PyNum& n);    // no '.', 'e', 'E'
double numFloat(const PyNum& n);  // float(text)

std::string pyStr(const PyValue& v);         // Python str(v)
std::string pyRepr(const PyValue& v);        // Python repr(v)
double pyNumberToFloat(const PyValue& v);    // Python float(v); throws
std::string pyTypeName(const PyValue& v);

// json.dumps(v, indent=2): 2-space indent, ",\n" item separator, ": " key
// separator, ensure_ascii escaping, NaN/Infinity literals, insertion order.
std::string pyJSONDumps(const PyValue& v);

// Go's %q for strings (error-message parity).
std::string goQuote(const std::string& s);

// ---------------------------------------------------------------- run records

struct Run {
    std::string task;
    double outcome = 0.0;
    int order = 0;  // position in the file, tie-break for stable sorts
    std::shared_ptr<PyObj> factors;  // all fields except task/id, file order

    // The factor's raw value (Python r.get(factor)); nullptr when absent.
    const PyValue* factorValue(const std::string& factor) const {
        if (!factors) return nullptr;
        return factors->get(factor);
    }
    // Run factors in file order (task/outcome are not factors).
    std::vector<std::string> flatKeys() const {
        if (!factors) return {};
        return factors->keys();
    }
};

Run runFromDict(const PyObj& o, int order);  // throws std::runtime_error
std::vector<Run> runsFromArray(const std::vector<PyValue>& arr, const std::string& src);
std::vector<Run> loadRuns(const std::string& path);
std::vector<std::string> sortedStringSet(const std::vector<std::string>& keys);
std::string readFileOrThrow(const std::string& path);  // throws on IO error

}  // namespace evalsig
