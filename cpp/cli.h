// cli.h — command line interface, byte-compatible with the Python reference
// (and the Go port):
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
#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace evalsig {

// CLI entry: args without the program name, exit code returned. Throws
// nothing: runtime errors are caught here and reported like Go's recover.
int cliMain(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace evalsig
