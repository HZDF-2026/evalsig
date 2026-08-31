// Command evalsig is the Go port of the evalsig CLI.
package main

import (
	"os"

	"github.com/HZDF-2026/evalsig/internal/evalsig"
)

func main() {
	os.Exit(evalsig.Main(os.Args[1:], os.Stdout, os.Stderr))
}
