// pyjson.cpp — JSON parser, Python value semantics, and run records.
// Port of runs.go plus the JSON emitters of pystat.go.

#include "pyjson.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "pynum.h"

namespace evalsig {

namespace {

struct Parser {
    const char* p;
    const char* end;

    explicit Parser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    void skipWs() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }

    [[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

    char peek() const { return *p; }

    PyValue parseValue() {
        skipWs();
        if (p >= end) fail("unexpected end of JSON input");
        char c = peek();
        switch (c) {
        case '{': return parseObject();
        case '[': return parseArray();
        case '"': return PyValue(parseString());
        case 't':
            expect("true");
            return PyValue(true);
        case 'f':
            expect("false");
            return PyValue(false);
        case 'n':
            expect("null");
            return PyValue(std::monostate{});
        default:
            if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
            fail(std::string("invalid character '") + c +
                 "' looking for beginning of value");
        }
    }

    void expect(const char* lit) {
        size_t n = std::strlen(lit);
        if (static_cast<size_t>(end - p) < n || std::memcmp(p, lit, n) != 0) {
            fail(std::string("invalid character '") + peek() + "' in literal");
        }
        p += n;
    }

    PyValue parseObject() {
        p++;  // '{'
        auto o = std::make_shared<PyObj>();
        skipWs();
        if (p < end && peek() == '}') {
            p++;
            return PyValue(std::move(o));
        }
        while (true) {
            skipWs();
            if (p >= end || peek() != '"') {
                fail("invalid character looking for beginning of object key string");
            }
            std::string key = parseString();
            skipWs();
            if (p >= end || peek() != ':') fail("invalid character after object key");
            p++;
            PyValue v = parseValue();
            o->pairs.emplace_back(std::move(key), std::move(v));
            skipWs();
            if (p >= end) fail("unexpected end of JSON input");
            if (peek() == ',') {
                p++;
                continue;
            }
            if (peek() == '}') {
                p++;
                break;
            }
            fail(std::string("invalid character '") + peek() +
                 "' after object key:value pair");
        }
        return PyValue(std::move(o));
    }

    PyValue parseArray() {
        p++;  // '['
        std::vector<PyValue> arr;
        skipWs();
        if (p < end && peek() == ']') {
            p++;
            return PyValue(std::move(arr));
        }
        while (true) {
            arr.push_back(parseValue());
            skipWs();
            if (p >= end) fail("unexpected end of JSON input");
            if (peek() == ',') {
                p++;
                continue;
            }
            if (peek() == ']') {
                p++;
                break;
            }
            fail(std::string("invalid character '") + peek() + "' after array element");
        }
        return PyValue(std::move(arr));
    }

    std::string parseString() {
        p++;  // '"'
        std::string out;
        while (true) {
            if (p >= end) fail("unexpected end of JSON input");
            unsigned char c = static_cast<unsigned char>(*p);
            if (c == '"') {
                p++;
                return out;
            }
            if (c == '\\') {
                p++;
                if (p >= end) fail("unexpected end of JSON input");
                char e = *p++;
                switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t r = parseHex4();
                    if (r >= 0xD800 && r <= 0xDBFF) {
                        // leading surrogate: only a trailing surrogate may follow
                        if (end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                            const char* save = p;
                            p += 2;
                            uint32_t r2 = parseHex4();
                            if (r2 >= 0xDC00 && r2 <= 0xDFFF) {
                                r = 0x10000 + ((r - 0xD800) << 10) + (r2 - 0xDC00);
                            } else {
                                p = save;
                                r = 0xFFFD;
                            }
                        } else {
                            r = 0xFFFD;
                        }
                    } else if (r >= 0xDC00 && r <= 0xDFFF) {
                        r = 0xFFFD;
                    }
                    appendRune(out, r);
                    break;
                }
                default:
                    fail(std::string("invalid character '") + e +
                         "' in string escape code");
                }
                continue;
            }
            if (c < 0x20) {
                fail(std::string("invalid character '") + static_cast<char>(c) +
                     "' in string literal");
            }
            // Pass bytes through; emission re-decodes UTF-8 like Go's range.
            out += static_cast<char>(c);
            p++;
        }
    }

    uint32_t parseHex4() {
        if (end - p < 4) fail("invalid \\u escape in string");
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("invalid \\u escape in string");
        }
        return v;
    }

    static void appendRune(std::string& out, uint32_t r) {
        if (r < 0x80) {
            out += static_cast<char>(r);
        } else if (r < 0x800) {
            out += static_cast<char>(0xC0 | (r >> 6));
            out += static_cast<char>(0x80 | (r & 0x3F));
        } else if (r < 0x10000) {
            out += static_cast<char>(0xE0 | (r >> 12));
            out += static_cast<char>(0x80 | ((r >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (r & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (r >> 18));
            out += static_cast<char>(0x80 | ((r >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((r >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (r & 0x3F));
        }
    }

    PyValue parseNumber() {
        const char* start = p;
        if (p < end && *p == '-') p++;
        if (p >= end) fail("invalid number");
        if (peek() == '0') {
            p++;
        } else if (peek() >= '1' && peek() <= '9') {
            while (p < end && peek() >= '0' && peek() <= '9') p++;
        } else {
            fail(std::string("invalid character '") + peek() +
                 "' in numeric literal");
        }
        if (p < end && peek() == '.') {
            p++;
            if (p >= end || peek() < '0' || peek() > '9') {
                fail("invalid character after decimal point in numeric literal");
            }
            while (p < end && peek() >= '0' && peek() <= '9') p++;
        }
        if (p < end && (peek() == 'e' || peek() == 'E')) {
            p++;
            if (p < end && (peek() == '+' || peek() == '-')) p++;
            if (p >= end || peek() < '0' || peek() > '9') {
                fail("invalid character in exponent of numeric literal");
            }
            while (p < end && peek() >= '0' && peek() <= '9') p++;
        }
        return PyValue(PyNum{std::string(start, p)});
    }
};

std::string pyJSONFloat(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    return pyReprFloat(v);
}

void writeEscapedRune(std::string& b, uint32_t r) {
    if (r > 0xFFFF) {
        writeEscapedRune(b, 0xD800 + ((r - 0x10000) >> 10));
        writeEscapedRune(b, 0xDC00 + ((r - 0x10000) & 0x3FF));
        return;
    }
    char buf[8];
    std::snprintf(buf, sizeof buf, "\\u%04x", r);
    b += buf;
}

}  // namespace

std::string pyJSONString(const std::string& s) {
    std::string b;
    b += '"';
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            switch (c) {
            case '"': b += "\\\""; break;
            case '\\': b += "\\\\"; break;
            case '\n': b += "\\n"; break;
            case '\r': b += "\\r"; break;
            case '\t': b += "\\t"; break;
            case '\b': b += "\\b"; break;
            case '\f': b += "\\f"; break;
            default:
                if (c < 0x20 || c >= 0x7F) {
                    writeEscapedRune(b, c);
                } else {
                    b += static_cast<char>(c);
                }
            }
            i++;
            continue;
        }
        // Multi-byte UTF-8: decode one rune (invalid sequences -> U+FFFD, the
        // Go `for range` behavior for the same bytes).
        uint32_t r = 0xFFFD;
        int n = 1;
        if ((c & 0xE0) == 0xC0 && i + 1 < s.size() &&
            (static_cast<unsigned char>(s[i + 1]) & 0xC0) == 0x80) {
            r = (c & 0x1F) << 6 | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
            n = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size() &&
                   (static_cast<unsigned char>(s[i + 1]) & 0xC0) == 0x80 &&
                   (static_cast<unsigned char>(s[i + 2]) & 0xC0) == 0x80) {
            r = (c & 0x0F) << 12 | (static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6 |
                (static_cast<unsigned char>(s[i + 2]) & 0x3F);
            n = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < s.size() &&
                   (static_cast<unsigned char>(s[i + 1]) & 0xC0) == 0x80 &&
                   (static_cast<unsigned char>(s[i + 2]) & 0xC0) == 0x80 &&
                   (static_cast<unsigned char>(s[i + 3]) & 0xC0) == 0x80) {
            r = (c & 0x07) << 18 | (static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12 |
                (static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6 |
                (static_cast<unsigned char>(s[i + 3]) & 0x3F);
            n = 4;
        }
        writeEscapedRune(b, r);
        i += static_cast<size_t>(n);
    }
    b += '"';
    return b;
}

namespace {

void emitJSON(std::string& b, const PyValue& v, int depth) {
    auto indentN = [&b](int d) { b.append(static_cast<size_t>(d) * 2, ' '); };
    if (v.v.index() == 0) {  // monostate == null
        b += "null";
        return;
    }
    if (const bool* t = std::get_if<bool>(&v.v)) {
        b += *t ? "true" : "false";
        return;
    }
    if (const long long* t = std::get_if<long long>(&v.v)) {
        b += std::to_string(*t);
        return;
    }
    if (const double* t = std::get_if<double>(&v.v)) {
        b += pyJSONFloat(*t);
        return;
    }
    if (const PyNum* t = std::get_if<PyNum>(&v.v)) {
        if (numIsInt(*t)) {
            b += t->text;
        } else {
            b += pyJSONFloat(numFloat(*t));
        }
        return;
    }
    if (const std::string* t = std::get_if<std::string>(&v.v)) {
        b += pyJSONString(*t);
        return;
    }
    if (const auto* arr = std::get_if<std::vector<PyValue>>(&v.v)) {
        if (arr->empty()) {
            b += "[]";
            return;
        }
        b += "[\n";
        for (size_t i = 0; i < arr->size(); i++) {
            if (i > 0) b += ",\n";
            indentN(depth + 1);
            emitJSON(b, (*arr)[i], depth + 1);
        }
        b += "\n";
        indentN(depth);
        b += "]";
        return;
    }
    if (const auto* o = std::get_if<std::shared_ptr<PyObj>>(&v.v)) {
        const PyObj& obj = **o;
        if (obj.pairs.empty()) {
            b += "{}";
            return;
        }
        b += "{\n";
        for (size_t i = 0; i < obj.pairs.size(); i++) {
            if (i > 0) b += ",\n";
            indentN(depth + 1);
            b += pyJSONString(obj.pairs[i].first);
            b += ": ";
            emitJSON(b, obj.pairs[i].second, depth + 1);
        }
        b += "\n";
        indentN(depth);
        b += "}";
        return;
    }
    throw std::runtime_error("pyJSONDumps: unsupported type");
}

}  // namespace

PyValue parsePyJSON(const std::string& data) {
    Parser ps(data);
    PyValue v = ps.parseValue();
    ps.skipWs();
    if (ps.p != ps.end) {
        throw std::runtime_error("extra data after JSON document");
    }
    return v;
}

bool numIsInt(const PyNum& n) {
    return n.text.find_first_of(".eE") == std::string::npos;
}

double numFloat(const PyNum& n) {
    errno = 0;
    double d = std::strtod(n.text.c_str(), nullptr);
    if (errno == ERANGE && d != 0.0) {
        // Go's ParseFloat returns ±Inf with an ErrRange error; the callers of
        // numFloat only surface the value, so clamp to the parsed result.
    }
    return d;
}

std::string pyStr(const PyValue& v) {
    if (v.v.index() == 0) return "None";
    if (const bool* t = std::get_if<bool>(&v.v)) return *t ? "True" : "False";
    if (const std::string* t = std::get_if<std::string>(&v.v)) return *t;
    if (const PyNum* t = std::get_if<PyNum>(&v.v)) {
        if (numIsInt(*t)) return t->text;
        return pyReprFloat(numFloat(*t));
    }
    return pyRepr(v);
}

std::string pyRepr(const PyValue& v) {
    if (v.v.index() == 0) return "None";
    if (const bool* t = std::get_if<bool>(&v.v)) return *t ? "True" : "False";
    if (const std::string* t = std::get_if<std::string>(&v.v)) {
        std::string out = *t;
        size_t pos = 0;
        while ((pos = out.find('\'', pos)) != std::string::npos) {
            out.replace(pos, 1, "\\'");
            pos += 2;
        }
        return "'" + out + "'";
    }
    if (const PyNum* t = std::get_if<PyNum>(&v.v)) return pyStr(PyValue(*t));
    if (const auto* arr = std::get_if<std::vector<PyValue>>(&v.v)) {
        std::string out = "[";
        for (size_t i = 0; i < arr->size(); i++) {
            if (i > 0) out += ", ";
            out += pyRepr((*arr)[i]);
        }
        return out + "]";
    }
    if (const auto* o = std::get_if<std::shared_ptr<PyObj>>(&v.v)) {
        std::string out = "{";
        const auto& pairs = (*o)->pairs;
        for (size_t i = 0; i < pairs.size(); i++) {
            if (i > 0) out += ", ";
            out += pyRepr(PyValue(pairs[i].first)) + ": " + pyRepr(pairs[i].second);
        }
        return out + "}";
    }
    if (const long long* t = std::get_if<long long>(&v.v)) return std::to_string(*t);
    if (const double* t = std::get_if<double>(&v.v)) return pyReprFloat(*t);
    return "?";
}

double pyNumberToFloat(const PyValue& v) {
    if (const bool* t = std::get_if<bool>(&v.v)) return *t ? 1.0 : 0.0;
    if (const PyNum* t = std::get_if<PyNum>(&v.v)) return numFloat(*t);
    if (const std::string* t = std::get_if<std::string>(&v.v)) {
        // Go: strconv.ParseFloat(strings.TrimSpace(t), 64)
        std::string s = *t;
        size_t b = s.find_first_not_of(" \t\n\r\v\f");
        size_t e = s.find_last_not_of(" \t\n\r\v\f");
        s = (b == std::string::npos) ? "" : s.substr(b, e - b + 1);
        if (s.empty()) {
            throw std::runtime_error("could not convert string to float: " + goQuote(*t));
        }
        const char* cs = s.c_str();
        char* endp = nullptr;
        errno = 0;
        double d = std::strtod(cs, &endp);
        if (endp != cs + s.size() || (d == 0.0 && errno == ERANGE && s != "0" &&
                                      s.find_first_not_of("-0.") != std::string::npos)) {
            throw std::runtime_error("could not convert string to float: " + goQuote(*t));
        }
        return d;
    }
    throw std::runtime_error("must be a real number, not " + pyTypeName(v));
}

std::string pyTypeName(const PyValue& v) {
    if (v.v.index() == 0) return "NoneType";
    if (std::get_if<bool>(&v.v)) return "bool";
    if (std::get_if<std::string>(&v.v)) return "str";
    if (std::get_if<PyNum>(&v.v)) return "int or float";
    if (std::get_if<std::vector<PyValue>>(&v.v)) return "list";
    if (std::get_if<std::shared_ptr<PyObj>>(&v.v)) return "dict";
    if (std::get_if<long long>(&v.v)) return "int or float";
    if (std::get_if<double>(&v.v)) return "int or float";
    return "unknown";
}

std::string pyJSONDumps(const PyValue& v) {
    std::string b;
    emitJSON(b, v, 0);
    return b;
}

std::string goQuote(const std::string& s) {
    std::string out = "\"";
    for (char ch : s) {
        unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\a': out += "\\a"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\v': out += "\\v"; break;
        default:
            if (c < 0x20 || c == 0x7F) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\x%02x", c);
                out += buf;
            } else {
                out += ch;
            }
        }
    }
    return out + "\"";
}

// ---------------------------------------------------------------- run records

namespace {

// Python _outcome probe order.
const char* outcomeKeys[] = {"success", "resolved", "passed", "score", "value", "outcome"};

double outcomeOf(const PyObj& o) {
    for (const char* k : outcomeKeys) {
        if (const PyValue* v = o.get(k)) {
            return pyNumberToFloat(*v);
        }
    }
    std::string keys = "[";
    auto ks = o.keys();
    for (size_t i = 0; i < ks.size(); i++) {
        if (i > 0) keys += " ";
        keys += ks[i];
    }
    keys += "]";
    throw std::runtime_error(
        "run has no outcome field (looked for success/resolved/passed/score/value): " +
        keys);
}

}  // namespace

std::string readFileOrThrow(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(path + ": open " + goQuote(path) +
                                 ": no such file or directory");
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Run runFromDict(const PyObj& o, int order) {
    std::string task;
    if (const PyValue* v = o.get("task")) {
        task = pyStr(*v);
    } else if (const PyValue* v = o.get("id")) {
        task = pyStr(*v);
    }
    double outcome = outcomeOf(o);
    auto f = std::make_shared<PyObj>();
    for (const auto& kv : o.pairs) {
        if (kv.first == "task" || kv.first == "id") continue;
        f->pairs.push_back(kv);
    }
    Run r;
    r.task = std::move(task);
    r.outcome = outcome;
    r.order = order;
    r.factors = std::move(f);
    return r;
}

std::vector<Run> runsFromArray(const std::vector<PyValue>& arr, const std::string& src) {
    std::vector<Run> runs;
    runs.reserve(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
        const auto* o = std::get_if<std::shared_ptr<PyObj>>(&arr[i].v);
        if (!o) {
            throw std::runtime_error(src + ": run " + std::to_string(i) +
                                     " is not a JSON object");
        }
        runs.push_back(runFromDict(**o, static_cast<int>(i)));
    }
    return runs;
}

std::vector<Run> loadRuns(const std::string& path) {
    std::string data;
    try {
        data = readFileOrThrow(path);
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    PyValue v;
    try {
        v = parsePyJSON(data);
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
    const auto* arr = std::get_if<std::vector<PyValue>>(&v.v);
    if (!arr) {
        throw std::runtime_error(path + ": expected a JSON array of runs");
    }
    return runsFromArray(*arr, path);
}

std::vector<std::string> sortedStringSet(const std::vector<std::string>& keys) {
    std::vector<std::string> out;
    for (const auto& k : keys) {
        if (std::find(out.begin(), out.end(), k) == out.end()) out.push_back(k);
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace evalsig
